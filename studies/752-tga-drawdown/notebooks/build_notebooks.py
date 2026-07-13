"""Generate the two narrative notebooks for Study 752 (TGA-Drawdown).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the hardcoded TGA
proxy (always available) and the cached SPY prices under ../_cache/, and otherwise quote
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


# Frozen real-tape headline numbers — mirror of docs/results.md (TGA monthly proxy of FRED
# WTREGEN + SPY month-end, 2005-01 -> 2026-06, 258 months, 21.4 years).
R = dict(
    start="2005-01-31", end="2026-06-30", months=258, years=21.4,
    # per-horizon: (months, n_draw, draw%, build%, base%, draw_up%, base_up%, welch_t, hac_t, p_placebo)
    h1=(1, 123, 0.91, 1.01, 0.96, 66, 66, -0.11, -0.44, 0.564),
    h2=(2, 123, 2.13, 1.79, 1.95, 76, 72, 0.27, -0.06, 0.379),
    h3=(3, 122, 3.58, 2.34, 2.93, 75, 74, 0.85, 0.79, 0.163),
    h6=(6, 120, 6.07, 5.77, 5.91, 82, 79, 0.13, -0.37, 0.448),
    # HAC (Newey-West) regression per horizon: beta%/+100B, hac_t, r2%
    reg={1: (-0.11, -0.44, 0.06), 2: (-0.03, -0.06, 0.00),
         3: (0.47, 0.79, 0.41), 6: (-0.29, -0.37, 0.07)},
    # lead/lag: L -> corr
    leadlag={-6: -0.070, -5: 0.133, -4: 0.053, -3: 0.080, -2: -0.083, -1: 0.045,
             0: 0.090, 1: -0.025, 2: 0.022, 3: 0.113, 4: 0.089, 5: -0.097, 6: -0.181},
    # overlay: (bh_mean%, bh_sharpe, gross_mean%, gross_sharpe, net_mean%, net_sharpe, switches)
    overlay=(11.6, 0.78, 7.4, 0.72, 6.7, 0.66, 147),
    # robustness 1m: (label, n_draw, draw1%, base1%, welch_t, hac_t)
    robust=[("k=1", 123, 0.91, 0.96, -0.11, -0.44), ("k=2", 113, 0.92, 0.96, -0.09, -0.02),
            ("k=3", 104, 1.58, 0.96, 1.36, 0.75), ("ex-COVID", 113, 0.87, 1.05, -0.34, 0.24)],
    # synthetic control: (edge, n_draw, draw1%, base1%, welch_t, hac_t, p)
    syn=[(0.0, 128, 0.74, 0.54, 0.44, 1.44, 0.294), (0.04, 128, 4.85, 2.64, 4.40, 7.45, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Hidden_liquidity_lever%3F: Not_supported](https://img.shields.io/badge/Hidden_liquidity_lever%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from tga_drawdown import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("SPY cache present:", HAVE_REAL,
      "| TGA+SPY months:", (0 if F is None else len(F)))
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
            "# When Uncle Sam spends down his checking account, do stocks go up? 🏦\n"
            "### The 'hidden liquidity lever' everyone on finance Twitter watches, in plain English\n\n"
            + BADGES +
            "The U.S. Treasury keeps its cash in a checking account at the Federal Reserve called the "
            "**Treasury General Account**, or **TGA**. When the Treasury *spends down* that account, the "
            "money lands in ordinary bank accounts — and a popular macro story says that flood of cash "
            "(a **liquidity injection**) quietly lifts the stock market over the following weeks. When the "
            "Treasury *refills* the account, the story runs in reverse: cash gets vacuumed out of the "
            "system and stocks are supposed to struggle. It's the market's favourite piece of "
            "plumbing.\n\n"
            "It's a great story. It's also testable. This notebook asks three blunt questions: when the "
            "TGA drains, does the market really do better next? Does the drain actually come **first** "
            "(that's the whole pitch)? And if you *bought* every time the TGA was draining, would you "
            "make money?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West regression and the "
            "synthetic control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The live FRED feed is blocked here, and the real TGA series is "
            "*weekly* (a thousand-plus points). Rather than fake weekly precision, we use a **labelled "
            "monthly proxy** of the balance — the big moves are faithful (the 2020 surge to ~$1.8 "
            "trillion, the debt-ceiling drains to near-zero), the exact levels are approximate. Every "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When the TGA drains, does the market do better next month? | **No — if anything, worse.** "
            f"The month after a drawdown SPY averages **+{R['h1'][2]:.2f}%** vs **+{R['h1'][4]:.2f}%** "
            "normally. The sign is the *opposite* of the story. |\n"
            "| Is there any horizon where it works? | **Not convincingly.** The best is a mild bump at "
            f"3 months (**+{R['h3'][2]:.1f}%** vs **+{R['h3'][4]:.1f}%**), but it's easily inside the "
            "noise — statistically you can't tell it from luck. |\n"
            "| Does the drain come *first*? | **No.** The injection lines up with market moves scattered "
            "randomly before and after it — at the horizons a real 'lever' needs, the link is essentially "
            "**zero**. It co-wanders; it doesn't lead. |\n"
            "| So could you trade it? | **It loses.** \"Buy when the TGA drains\" earned "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%** for just holding — you give "
            "up return for a lever that isn't there. |\n\n"
            "> The plumbing is real. But \"a TGA drawdown lifts stocks\" is a tidy narrative the tape "
            "doesn't back — largely because the *biggest* drains are debt-ceiling emergencies, not "
            "stimulus."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Treasury General Account is a hidden liquidity lever. When the Treasury draws it "
            "down, that cash becomes bank reserves — a stealth injection that lifts risk assets over the "
            "following weeks. When it rebuilds the account, reserves drain and stocks come under "
            "pressure. Watch the TGA and you're watching the market's fuel gauge.\"*\n\n"
            "There's a respectable backbone to this: the TGA really *is* one of the big swing factors for "
            "the level of bank reserves, and reserves are the headline term in the 'net liquidity' "
            "framework (*Fed balance sheet − TGA − reverse repos*) that macro strategists like Michael "
            "Howell popularised. So the mechanism isn't crazy. The trading leap is the part we test: "
            "that a TGA *drawdown* arrives early enough, and cleanly enough, to be a **tradable** signal "
            "for equities."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be a gift: a free, publicly-reported cash balance that tells you when to "
            "lean *into* stocks. But 'liquidity' hides two traps. First, the TGA doesn't move reserves "
            "one-for-one — the reverse-repo facility often soaks up the difference, so the pass-through "
            "is contingent, not automatic. Second, and worse for the story: the **biggest** TGA "
            "drawdowns in modern history are **debt-ceiling standoffs** (2021, 2023, 2025), when the "
            "Treasury runs its cash to zero *because it legally can't borrow* — a stressed, uncertain "
            "regime, not a stimulative one. So the very moments the gauge screams 'injection!' are the "
            "moments a liquidity bull should be most nervous. That's the confound to keep in mind."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f} years** ({R['start'][:4]}–{R['end'][:4]}, "
            f"{R['months']} months) of the TGA balance against month-end SPY, and:\n\n"
            "1. **Split the months.** Call the TGA **drawing down** when this month's balance is below "
            "last month's. Compare what SPY did next (1/2/3/6 months) in drawdown months vs all "
            "months.\n"
            "2. **Check the timing.** The crucial test: slide the injection forward and backward against "
            "the market and find *where* they line up best. If the drain truly **leads**, the strongest "
            "*positive* link shows up at a **positive lead** (drain first, rally later).\n"
            "3. **Try to trade it.** Hold SPY whenever the TGA is draining, sit in cash when it's "
            "refilling, pay realistic costs — and see if it beats just buying and holding.\n\n"
            "**What would make us say 'mirage'?** If the next-month edge is inside the noise, the "
            "injection shows no clean lead, and the buy-when-draining rule loses to buy-and-hold — then "
            "the lever is a story, not a signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw material.** Here's the TGA balance over two decades — sleepy and small "
            "before 2015, then the enormous COVID-2020 balloon toward ~$1.8 trillion, and the "
            "debt-ceiling drains that repeatedly take it to near-zero. The TGA clearly *does* big, "
            "macro-relevant things. The question is whether those things help you time stocks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = F['tga']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(g.index, g.values, c=GREEN, lw=1.4)\n"
            "    ax.set_yscale('log')\n"
            "    ax.set_title('Treasury General Account balance ($B, log scale) — monthly proxy')\n"
            "    ax.set_ylabel('TGA balance ($B)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('peak TGA:', int(g.max()), '$B, around', g.idxmax().date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md; TGA proxy peaked ~1750 $B mid-2020')"
        ),
        md(
            "**Now the payoff.** For each horizon, the average forward SPY return in **TGA-draining** "
            "months next to the return on an **average** month. The folklore predicts the green bars sit "
            "*above* the grey ones."
        ),
        code(
            "hs = [1, 2, 3, 6]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, m) for m in hs]\n"
            "    dr = [r['draw_mean']*100 for r in rows]; base = [r['base_mean']*100 for r in rows]\n"
            "else:\n"
            "    dr = [R['h1'][2], R['h2'][2], R['h3'][2], R['h6'][2]]\n"
            "    base = [R['h1'][4], R['h2'][4], R['h3'][4], R['h6'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, dr, .4, color=GREEN, label='after TGA DRAWS DOWN')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='an average month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('average forward SPY return (%)')\n"
            "ax.set_title('TGA drawdown -> higher forward returns? Barely, and not at 1 month')\n"
            "for i,(a,b) in enumerate(zip(dr,base)):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('1-month: drawdown', f'{dr[0]:.2f}%', 'vs base', f'{base[0]:.2f}%')"
        ),
        md(
            f"Look at the **1-month** bars — the horizon the 'following weeks' pitch cares about most. "
            f"The drawdown average (**+{R['h1'][2]:.2f}%**) is actually *below* the base rate "
            f"(**+{R['h1'][4]:.2f}%**). The sign is **backwards**. A small positive gap shows up only at "
            f"3 months (**+{R['h3'][2]:.1f}%** vs **+{R['h3'][4]:.1f}%**), and even that is a rounding "
            "error's worth of edge. Hold that thought; the *next* chart is where the mechanism is "
            "supposed to show up — and doesn't."
        ),
        md(
            "**The crucial test: does the TGA drain come *first*?** We slide the injection forward and "
            "backward against the market and measure how tightly they move together. A real lever would "
            "show its strongest *positive* link at a **positive lead** (injection leads → bar rises on "
            "the right). Watch where it actually peaks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F)\n"
            "    Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [GREEN if L>0 else GREY for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=RED, lw=1, ls=':')\n"
            "ax.set_xlabel('lead L (months): L>0 = injection moves FIRST (a real lever)   |   L<0 = injection LAGS the market')\n"
            "ax.set_ylabel('correlation with market move'); ax.set_xticks(Ls)\n"
            "ax.set_title('No coherent peak on the right: the injection does not lead the market')\n"
            "plt.tight_layout(); plt.show()\n"
            "imax = int(np.nanargmax(cs))\n"
            "print(f'strongest POSITIVE link at L={Ls[imax]} months; corr at +1 month = {cs[Ls.index(1)]:+.2f}')"
        ),
        md(
            "There's no there there. The strongest *positive* bar is at **L = −5** — the injection lining "
            "up with a market move that happened months *earlier* — and at exactly the positive leads a "
            "real lever needs (**+1, +2 months**) the correlation is basically **zero** (even slightly "
            "*negative* at +1). The bars scatter around zero with no pattern. **The TGA drain doesn't "
            "lead the market**; it wanders alongside it, sharing the same macro weather."
        ),
        md(
            "**Could you trade it anyway?** Suppose you held SPY every month the TGA was draining and sat "
            "in cash otherwise. Here's that strategy's growth vs just buying and holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    draw = st.drawdown_mask(F); pos = draw.astype(float).shift(1)\n"
            "    rr = F['spy'].pct_change()\n"
            "    import pandas as pd\n"
            "    dfp = pd.DataFrame({'r': rr, 'pos': pos}).dropna()\n"
            "    sw = dfp['pos'].diff().abs().fillna(0); c=10/1e4\n"
            "    overlay = (dfp['pos']*dfp['r'] - sw*c)\n"
            "    bh_grow = (1+dfp['r']).cumprod(); ov_grow = (1+overlay).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(bh_grow.index, bh_grow.values, c=GREY, lw=1.8, label='buy & hold SPY')\n"
            "    ax.plot(ov_grow.index, ov_grow.values, c=RED, lw=1.8, label='hold when TGA draining (net)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('\"Buy when the TGA drains\" trails buy-and-hold')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> buy&hold {bh_grow.iloc[-1]:.1f}x  vs  overlay {ov_grow.iloc[-1]:.1f}x')\n"
            "else:\n"
            "    print(f\"overlay {R['overlay'][4]:.1f}%/yr vs buy-hold {R['overlay'][0]:.1f}%/yr (net) — see results.md\")"
        ),
        md(
            f"The overlay ends up **below** buy-and-hold — **+{R['overlay'][4]:.1f}%/yr** vs "
            f"**+{R['overlay'][0]:.1f}%/yr** net, and a *lower* Sharpe ({R['overlay'][5]:.2f} vs "
            f"{R['overlay'][1]:.2f}). Every time you sat out because the TGA was *building*, you often sat "
            "out a rally — the account rebuilt through big chunks of the 2016–2019 and 2023–2024 bull "
            "runs. The signal doesn't just fail to help; **it costs you.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The next-month sign is *backwards*, no horizon clears the significance "
            "bar, and a Newey-West regression finds a slope of essentially zero. Indistinguishable from "
            "noise.\n"
            "- **Tradability — Mirage.** Buying on TGA drawdowns **loses to buy-and-hold**. Nothing to "
            "deploy.\n"
            "- **Hidden liquidity lever? — Not supported.** The injection shows no coherent lead over the "
            "market — near-zero exactly where a real lever would bite. The plumbing exists; the tradable "
            "lever doesn't."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Forget significance for a second. Even if there were a whisper of an edge, the operational "
            "reality kills it. The TGA drains in **about half** of all months (every wiggle down counts), "
            f"so a buy-on-draining rule flips in and out constantly — **{R['overlay'][6]} switches** in "
            "21 years — chewing costs for a signal that isn't pointing anywhere. And the confound is "
            "fatal: the deepest, most attention-grabbing drawdowns are **debt-ceiling emergencies**, "
            "when the Treasury is running on fumes because it can't issue debt — exactly the anxious "
            "regime a liquidity bull is wrong to buy. There's no version of \"buy when the TGA drains\" "
            "that both fires cleanly and makes money."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling test.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) "
            "asks the same question of the 'most important weekly number': real macro signal, but can you "
            "trade it?\n"
            "- **Do the full 'net liquidity' sum.** We used the TGA alone. The strategist version is "
            "*Fed balance sheet − TGA − reverse repos* — a Beat-7 fork is to build that composite and see "
            "if the reverse-repo term rescues the signal (spoiler from the plumbing: in 2022–23 the RRP "
            "absorbed much of the TGA move).\n"
            "- **Go weekly.** Our proxy is monthly; the thesis is really about *weekly* dynamics. Pull the "
            "true weekly `WTREGEN` and test 1–4 week horizons — a finer clock is the fair next test, "
            "though a monthly null this flat rarely turns into a weekly edge.\n\n"
            "*Think the TGA drawdown leads the market? Show the lead/lag chart peaking on the **right** "
            "(positive lead) — then we'll talk.*"
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
            "# TGA-Drawdown — a quantitative teardown 🔬\n"
            "### Drawdown-vs-build split returns · Welch *t* + Newey-West (HAC) predictive regression + "
            "placebo null · the decisive lead/lag cross-correlation · a timing overlay vs buy-and-hold · "
            "robustness · a synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "macro-liquidity thesis fuses two claims: that a TGA drawdown (1) **predicts** stronger "
            "equity returns and (2) does so **early** enough to trade. We separate them. The conditional "
            "tilt is *insignificant and wrong-signed at 1 month*; the continuous HAC regression finds a "
            "*near-zero slope*; and the **lead/lag structure** shows the injection is neither a clean "
            "leader nor a lagger — it co-wanders. A timing overlay that *underperforms* buy-and-hold "
            "seals the Tradability axis.\n\n"
            "> ⚠️ **Data + proxy note.** FRED's CSV endpoint is firewalled here and `WTREGEN` is *weekly*; "
            "the TGA tape is a hardcoded monthly **proxy** ($B) — landmark moves faithful, exact levels "
            "approximate (named on the Signal axis). SPY is yfinance daily adjusted close (total-return), "
            "month-end sampled. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | 1-month drawdown mean **+{R['h1'][2]:.2f}%** vs base "
            f"**+{R['h1'][4]:.2f}%** (*wrong sign*); best Welch **t = +{R['h3'][7]:.2f}** (3m), HAC slope "
            f"**t** never leaves ±1, R² ≤ 0.4%. Noise. |\n"
            f"| **Tradability** | `MIRAGE` | Hold-when-draining overlay **+{R['overlay'][4]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][5]:.2f}**) **vs buy-hold +{R['overlay'][0]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][1]:.2f}**). Acting on it destroys return. |\n"
            f"| **Hidden liquidity lever?** | `NOT SUPPORTED` | Lead/lag corr ≈ 0 at L=+1/+2 "
            f"(**{R['leadlag'][1]:+.2f} / {R['leadlag'][2]:+.2f}**); extremes scattered at L=−5, +6. No "
            "coherent lead. |\n\n"
            "> 💡 In plain words: the TGA→reserves pass-through is real *plumbing*, but it's mediated by "
            "the reverse-repo facility and swamped by debt-ceiling episodes (the biggest drawdowns are "
            "*forced*, not stimulative). A macro series that co-moves with the cycle need not **lead** a "
            "market that leads the cycle itself."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $g_t$ be the TGA balance ($B) and $d_t = g_t - g_{t-1}$ its 1-month change; the "
            "**injection** is $x_t = -d_t$ (positive = drawdown). The TGA is **DRAWING DOWN** at $t$ when "
            "$d_t < 0$. With a one-month execution lag (the month-$t$ balance is acted on at the close of "
            "$t+1$), define forward return $r_{t+1\\to t+1+H}$.\n\n"
            "- **H₁ (predicts).** $\\mathbb{E}[r\\mid \\text{drawdown}] > \\mathbb{E}[r]$ — a *positive* "
            "excess over the base rate; and the continuous slope $\\partial r/\\partial x > 0$.\n"
            "- **H₂ (leads).** The strongest *positive* injection↔return correlation sits at a "
            "**positive** lead (injection moves first).\n"
            "- **H₃ (deployable).** A hold-on-drawdown overlay beats buy-and-hold net of costs.\n\n"
            "We find **H₁ rejected** (1-month excess is *negative*; best $t=+0.85$; HAC slope $t\\in[-0.4,+0.8]$), "
            "**H₂ rejected** (no coherent lead; $\\rho\\approx0$ at $L=+1,+2$), **H₃ rejected** (overlay "
            "underperforms). The folklore is right about the *plumbing* (the TGA is a genuine reserves "
            "swing factor) and wrong about everything that would **pay** (a leading, tradable tilt)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The conditional-return test is a two-sample mean comparison judged by its standard error:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{draw}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{draw}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "The continuous version is a **Newey-West (HAC)** regression of the overlapping forward "
            "return on the injection, $r_{t+1\\to t+1+H} = \\alpha + \\beta\\,x_t + \\varepsilon$, with "
            "HAC lags $=H$ to absorb the overlap. But a significant $\\beta$ would **still not** establish "
            "*leading*: a coincident or lagging series can co-move with forward returns through cycle "
            "autocorrelation. The identifying test is the **lead/lag cross-correlation** "
            "$\\rho(L) = \\mathrm{corr}(x_t,\\ r_{t+L\\to t+L+1})$. A genuine lever peaks (positively) at "
            "$L>0$. If $\\rho(L)$ has no coherent positive-$L$ peak, the 'lever' is co-movement, not "
            "causation you can front-run."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **TGA tape.** Monthly proxy of `WTREGEN` ($B), {R['start'][:7]}→{R['end'][:7]} "
            f"({R['months']} months). Approximate levels, faithful landmarks (named on the axis).\n"
            "- **Signal.** $x_t = -(g_t - g_{t-1})$; DRAWING DOWN when $g_t < g_{t-1}$.\n"
            "- **Forward returns.** Enter at the close **1 month after** the signal (no look-ahead), "
            "hold $H\\in\\{1,2,3,6\\}$ months; drop horizons that overrun the tape.\n"
            "- **Null #1 (Welch t).** Drawdown-set mean vs the unconditional mean.\n"
            "- **Null #2 (HAC regression).** Newey-West slope of forward return on the injection, "
            "$\\text{lags}=H$ — the headline inference for the *dose-response*.\n"
            "- **Null #3 (placebo).** 20,000 draws of $k$ random months; "
            "$p = \\Pr[\\text{random-draw mean} \\ge \\text{drawdown mean}]$ (as bullish or more).\n"
            "- **Identification (lead/lag).** $\\rho(L)$ for $L\\in[-6,6]$ — *where* does the injection "
            "line up?\n"
            "- **Tradability.** Hold-on-drawdown overlay, 1-month lag, 10 bps one-way per switch "
            "(turnover one-way × NAV), excess-of-zero Sharpe (cash leg = 0, labelled).\n"
            "- **Positive control.** A deterministic series with a *planted* drawdown→returns link in the "
            "**capturable** ($t{+}1$-lagged) window: `edge=0` must not fake significance; a large `edge` "
            "must light up both the Welch and HAC tests."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — wrong-signed at 1m, insignificant everywhere\n\n"
            "Drawdown-set forward mean with $\\pm$ standard error against the unconditional base rate "
            "(dashed). At 1 month the drawdown mean is *below* base; where it edges above, the SE swamps "
            "the gap."
        ),
        code(
            "hs = [1, 2, 3, 6]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts, ses = [], [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize(F, m); cm.append(s['draw_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        d,_b,_a = st.split_returns(F, m); ses.append(d.std(ddof=1)/np.sqrt(len(d)))\n"
            "else:\n"
            "    cm = [R['h1'][2]/100, R['h2'][2]/100, R['h3'][2]/100, R['h6'][2]/100]\n"
            "    bm = [R['h1'][4]/100, R['h2'][4]/100, R['h3'][4]/100, R['h6'][4]/100]\n"
            "    ts = [R['h1'][7], R['h2'][7], R['h3'][7], R['h6'][7]]; ses = [.011,.017,.021,.03]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='drawdown (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('At 1m the drawdown bar sits BELOW base; the SE swamps every gap'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 1m the drawdown mean is **+{R['h1'][2]:.2f}%** vs base "
            f"**+{R['h1'][4]:.2f}%** — the *wrong sign* (Welch **t = {R['h1'][7]:.2f}**). The only "
            f"positive gap is at 3m (**t = +{R['h3'][7]:.2f}**), still far under 2. H₁'s split test is "
            "**rejected**: no horizon delivers a significant positive tilt, and the flagship horizon "
            "leans the wrong way."
        ),
        md(
            "### 4b · The continuous dose-response — HAC (Newey-West) slope ≈ 0\n\n"
            "Regress the forward $H$-month return on the injection (per +\\$100B drawn down), Newey-West "
            "HAC lags $=H$. If the lever is real, more drawdown ⇒ higher return ⇒ a *positive*, "
            "significant $\\beta$."
        ),
        code(
            "hs = [1, 2, 3, 6]\n"
            "if HAVE_REAL:\n"
            "    betas, thac, r2s = [], [], []\n"
            "    for m in hs:\n"
            "        rr = st.hac_regression(F, m); betas.append(rr['beta']*100); thac.append(rr['t_hac']); r2s.append(rr['r2']*100)\n"
            "else:\n"
            "    betas = [R['reg'][m][0] for m in hs]; thac = [R['reg'][m][1] for m in hs]; r2s = [R['reg'][m][2] for m in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, (a1,a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "cols = [GREEN if t>0 else RED for t in thac]\n"
            "a1.bar(x, thac, color=cols, width=.55)\n"
            "a1.axhline(2, ls='--', c=GREEN, label='+2 (significance bar)'); a1.axhline(-2, ls='--', c=RED)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(x); a1.set_xticklabels([f'{m}m' for m in hs])\n"
            "a1.set_ylabel('HAC (Newey-West) t on injection'); a1.set_ylim(-2.4, 2.4)\n"
            "for i,t in enumerate(thac): a1.annotate(f'{t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "a1.set_title('Slope t never leaves +/-1'); a1.legend(fontsize=8)\n"
            "a2.bar(x, r2s, color=GREY, width=.55); a2.set_xticks(x); a2.set_xticklabels([f'{m}m' for m in hs])\n"
            "a2.set_ylabel('regression R^2 (%)'); a2.set_ylim(0, 1.0)\n"
            "for i,v in enumerate(r2s): a2.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_title('Injection explains ~0% of return variance')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HAC beta (%/+100B), t, R2%:', [(round(b,2), round(t,2), round(r,2)) for b,t,r in zip(betas,thac,r2s)])"
        ),
        md(
            f"> 💡 In plain words: the slope **flips sign** across horizons "
            f"(**{R['reg'][1][0]:+.2f}%** at 1m, **{R['reg'][3][0]:+.2f}%** at 3m, "
            f"**{R['reg'][6][0]:+.2f}%** at 6m per +\\$100B) with HAC $t$ inside $\\pm1$ and R² ≤ 0.4%. "
            "There is **no dose-response**: a bigger drawdown does not buy a bigger return. H₁'s "
            "continuous form is **rejected** too."
        ),
        md(
            "### 4c · The decisive identification test — lead/lag\n\n"
            "$\\rho(L) = \\mathrm{corr}(x_t, r_{t+L\\to t+L+1})$. A real lever dips (rises) on the "
            "**right** (positive $L$, injection leads); random co-movement scatters around zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [GREEN if L>0 else GREY for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=RED, lw=1, ls=':')\n"
            "i1 = Ls.index(1)\n"
            "ax.annotate('a real lever peaks HERE\\n(positive lead) — it does not', xy=(1, cs[i1]),\n"
            "            xytext=(2.2, 0.16), ha='center', color=RED,\n"
            "            arrowprops=dict(arrowstyle='->', color=RED))\n"
            "ax.set_xlabel('lead L (months): L>0 = injection leads (a real lever)   |   L<0 = injection lags')\n"
            "ax.set_ylabel(r'$\\rho(L)$'); ax.set_xticks(Ls)\n"
            "ax.set_title('No coherent positive-L peak: injection co-wanders, it does not lead')\n"
            "plt.tight_layout(); plt.show()\n"
            "imax = int(np.nanargmax(cs))\n"
            "print(f'argmax rho at L={Ls[imax]} (rho={cs[imax]:+.2f}); rho at +1 = {cs[i1]:+.2f}, at +2 = {cs[Ls.index(2)]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: $\\arg\\max_L \\rho(L) = -5$ (injection lining up with a *past* market "
            f"move), while at the positive leads a lever needs — $L=+1,+2$ — "
            f"$\\rho = {R['leadlag'][1]:+.2f}, {R['leadlag'][2]:+.2f} \\approx 0$. The cross-correlation "
            "is structureless noise. **H₂ rejected.** This is the load-bearing result: the TGA injection "
            "does not lead equities, independent of the (already-insignificant) conditional mean."
        ),
        md(
            "### 4d · Tradability — the hold-on-drawdown overlay loses\n\n"
            "Hold SPY when the TGA is drawing down, cash when building (1-month lag, 10 bps/switch). "
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
            "a1.set_ylabel('annualised mean return (%)'); a1.set_title('Return: overlay loses ~5 pts/yr')\n"
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
            "the TGA *builds* through large stretches of bull markets (post-2016, post-2023), sitting out "
            "on 'building' forfeits upside. **H₃ rejected** — costs aren't even the issue; the *timing* "
            "loses. `MIRAGE`."
        ),
        md(
            "### 4e · Robustness — window and the COVID dependence\n\n"
            "Vary the change window $k$ and drop the COVID balloon-and-drawdown (2020–mid-2021). The "
            "1-month *t* never clears 2, and dropping COVID leaves it *negative*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for k in (1,2,3):\n"
            "        s = st.summarize(F, 1, k=k); rob.append((f'k={k}', s['n_draw'], s['draw_mean']*100, s['t'], s['t_hac']))\n"
            "    F2 = F[(F.index < '2020-01-01') | (F.index >= '2021-07-01')]\n"
            "    s = st.summarize(F2, 1); rob.append(('ex-COVID', s['n_draw'], s['draw_mean']*100, s['t'], s['t_hac']))\n"
            "else:\n"
            "    rob = [(l,n,d,wt,ht) for (l,n,d,_b,wt,ht) in R['robust']]\n"
            "labels = [r[0] for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "cols = [GREEN if t>0 else RED for t in tt]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(labels, tt, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=GREEN, label='t=+2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): ax.annotate(f'n={k}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('Welch t (1-month)'); ax.set_ylim(-1, 2.4)\n"
            "ax.set_title('No window clears |t|=2; ex-COVID flips negative'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, n, draw1%, Welch_t, HAC_t):', [(r[0], r[1], round(r[2],2), round(r[3],2), round(r[4],2)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: the best reading is the slow **k=3** window "
            f"(**Welch t={R['robust'][2][4]:.2f}**) — still under 2, and a 3-month-old drawdown trend is "
            "a weaker version of the 'immediate injection' the thesis sells. Drop COVID and the 1-month "
            f"*t* is **{R['robust'][3][4]:.2f}** — the *wrong sign*. No specification rescues a signal."
        ),
        md(
            "### 4f · Faithful-engine control — we know the truth here\n\n"
            "A deterministic monthly series with a *planted* link (a drawdown at $t$ lifts the "
            "**capturable** $[t{+}1,t{+}2]$ forward return by `edge`). With `edge=0` the test must stay "
            "flat; with a large `edge` both the Welch and HAC tests must light up — proving the engine is "
            "unbiased and the real-tape null isn't a measurement failure."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.04):\n"
            "    syn = data.synthetic_tga(n_months=252, edge=edge, seed=752)\n"
            "    s = st.summarize(syn, 1, k=1)\n"
            "    res.append((edge, s['n_draw'], s['draw_mean']*100, s['base_mean']*100, s['t'], s['t_hac'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / month' for e,*_ in res]\n"
            "wt = [r[4] for r in res]; ht = [r[5] for r in res]\n"
            "xb = np.arange(len(res))\n"
            "ax.bar(xb-.2, wt, .4, color=GREY, label='Welch t')\n"
            "ax.bar(xb+.2, ht, .4, color=GREEN, label='HAC t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=+2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(xb); ax.set_xticklabels(labels)\n"
            "for i,(a,b) in enumerate(zip(wt,ht)):\n"
            "    ax.annotate(f'{a:.2f}',(i-.2,a),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.2f}',(i+.2,b),ha='center',va='bottom')\n"
            "ax.set_ylabel('t-stat (1-month)'); ax.set_title('Control: no link -> flat; real link -> lights up'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,ht_,p in res: print(f'planted {e*100:+.0f}%/mo: n_draw={k} draw={c:.2f}% base={b:.2f}% Welch_t={t:.2f} HAC_t={ht_:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted link the headline Welch test sits at "
            f"**t = +{R['syn'][0][4]:.2f}** (placebo p={R['syn'][0][6]:.2f} — no false positive); a "
            f"**+4%/month** planted link drives **Welch t = +{R['syn'][1][4]:.2f}** and "
            f"**HAC t = +{R['syn'][1][5]:.2f}**. The machinery is honest and *does* bank a real "
            "drawdown→returns link — so the real-tape null (Welch/HAC $t\\approx0$) is a *genuine* absence "
            "of edge, not a broken test."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — 1m drawdown excess **{R['h1'][2]-R['h1'][4]:+.2f}pp** (*wrong sign*) "
            f"at Welch **t = {R['h1'][7]:.2f}**; best horizon (3m) only **t = +{R['h3'][7]:.2f}**; the HAC "
            "slope flips sign across horizons with $|t|<1$ and R² ≤ 0.4%. No conditional tilt and no "
            "dose-response — indistinguishable from noise.\n"
            f"- **Tradability `MIRAGE`** — the hold-on-drawdown overlay returns "
            f"**+{R['overlay'][4]:.1f}%/yr** (Sharpe {R['overlay'][5]:.2f}) vs buy-hold "
            f"**+{R['overlay'][0]:.1f}%/yr** (Sharpe {R['overlay'][1]:.2f}). Acting on the signal "
            "*subtracts* return — nothing to allocate to.\n"
            "- **Hidden liquidity lever? `NOT SUPPORTED`** — the lead/lag scan is structureless "
            f"($\\rho\\approx0$ at $L=+1,+2$; extremes at $L=-5,+6$). The injection co-wanders with a "
            "shared macro backdrop rather than leading the market. The mechanism the story rests on "
            "doesn't show up in the tape."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even a real tilt wouldn't deploy\n\n"
            "Grant the lore a whisper of an edge. The operational reality still defeats it. The TGA "
            "draws down in **~48% of months** (every down-wiggle of a noisy monthly series), so the "
            f"overlay churns ({R['overlay'][6]} switches / 21y) and is out of the market in roughly half "
            "of all months — including large stretches of bull runs when the Treasury was *rebuilding* "
            "cash. That structural mistiming is why the overlay's Sharpe is **below** passive even at "
            "zero cost. And the identification is fatal: the deepest drawdowns are **debt-ceiling** "
            "episodes — the Treasury spending its last cash because it *can't issue debt* — a stressed "
            "regime, not a stimulative one. The 'net liquidity' composite (subtracting the reverse-repo "
            "facility too) is the natural next test, but a monthly TGA null this flat, with no lead "
            "structure, is not a promising base to build on."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sibling.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/): a "
            "famous 'leading' macro series, same hardcoded-snapshot + SPY method — real signal, "
            "un-tradable timing.\n"
            "- **The full composite.** Build **net liquidity** = Fed balance sheet − TGA − RRP and rerun "
            "the identification test; the reverse-repo term is exactly what muted the 2022–23 TGA "
            "rebuild's predicted drain, so it's the fair steelman.\n"
            "- **Weekly resolution + real-time vintages.** Replace the monthly proxy with the true weekly "
            "`WTREGEN` (and DTS same-day operating balance) and test 1–4 week horizons with a proper "
            "VAR / Granger test; the coincident, no-lead structure is unlikely to survive into a lead at "
            "a finer clock, but it is the honest next experiment.\n\n"
            "*The reproducible core is offline and deterministic; the TGA input is an explicit labelled "
            "monthly proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
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
