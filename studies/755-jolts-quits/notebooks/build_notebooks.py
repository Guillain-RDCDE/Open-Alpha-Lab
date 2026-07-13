"""Generate the two narrative notebooks for Study 755 (JOLTS-Quits).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the hardcoded quits
snapshot (always available) and the cached SPY/XLY/XLP prices under ../_cache/, and
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


# Frozen real-tape headline numbers — mirror of docs/results.md (FRED JTSQUR hardcoded
# monthly snapshot + SPY/XLY/XLP month-end, 2000-12 -> 2026-05, 306 months, 25.4 years).
R = dict(
    start="2000-12-31", end="2026-05-31", months=306, years=25.4, frac_falling=29,
    # per-horizon SPY: (months, n_falling, falling%, rising%, base%, fall_down%, base_down%, t, p_placebo)
    h1=(1, 89, 0.62, 0.94, 0.84, 35, 35, -0.40, 0.308),
    h3=(3, 88, 2.10, 2.67, 2.51, 30, 29, -0.40, 0.302),
    h6=(6, 87, 5.65, 4.90, 5.12, 26, 25, 0.36, 0.679),
    h12=(12, 84, 11.39, 10.55, 10.79, 20, 19, 0.27, 0.635),
    # lead/lag: L -> corr(quits-mom@t, SPY ret over [t+L, t+L+1])
    leadlag={-6: 0.046, -5: 0.075, -4: 0.217, -3: 0.195, -2: 0.138, -1: -0.040,
             0: 0.056, 1: 0.077, 2: 0.031, 3: 0.030, 4: -0.004, 5: -0.032, 6: -0.059},
    # overlay: (bh_mean%, bh_sharpe, gross_mean%, gross_sharpe, net_mean%, net_sharpe, switches,
    #           bh_terminal, overlay_terminal)
    overlay=(9.7, 0.64, 8.1, 0.69, 7.8, 0.66, 90, 8.7, 6.0),
    # cyclicals (XLY-XLP) leg: (months, n_falling, falling%, base%, t, p)
    cyc=[(1, 89, 0.28, 0.34, -0.10, 0.451), (3, 88, 1.15, 0.92, 0.22, 0.616),
         (6, 87, 3.15, 1.72, 1.14, 0.917), (12, 84, 6.33, 3.64, 1.66, 0.969)],
    # robustness 12m: (label, n_falling, falling12%, base12%, t, p)
    robust=[("k=1", 85, 9.5, 10.8, -0.61, 0.227), ("k=3", 84, 11.4, 10.8, 0.27, 0.635),
            ("k=6", 97, 13.8, 10.8, 1.33, 0.971), ("thr>0.2pp", 19, 12.4, 10.8, 0.26, 0.660),
            ("ex-COVID", 80, 11.0, 11.8, -0.37, 0.321)],
    # lag sensitivity 12m: lag -> t
    lag={0: -0.25, 1: 0.10, 2: 0.27},
    # synthetic control: (edge, n_falling, falling1m%, base1m%, t, p)
    syn=[(0.0, 138, 1.48, 1.03, 1.15, 0.905), (0.04, 138, -1.92, -0.85, -2.64, 0.002)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Leading_gauge%3F: Not_supported](https://img.shields.io/badge/Leading_gauge%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from jolts_quits import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("SPY cache present:", HAVE_REAL,
      "| quits+SPY months:", (0 if F is None else len(F)),
      "| cyclicals:", data.have_cyclicals())
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
            "# When people stop quitting, is a market drop coming? 🧑‍🏭\n"
            "### The JOLTS 'quits rate' as a worker-confidence crystal ball, in plain English\n\n"
            + BADGES +
            "Every month the U.S. reports how many workers **voluntarily quit** their jobs — the JOLTS "
            "*quits rate*. The logic is lovely: you only quit when you're confident you'll land something "
            "better, so a **high** quits rate means workers feel great, and a **falling** one means "
            "confidence is draining away. And if workers are the first to sense trouble, a **drop in "
            "quits should warn you before stocks — especially economically-sensitive 'cyclical' stocks — "
            "roll over.**\n\n"
            "Great story. Testable story. This notebook asks three blunt questions: when quits fall, does "
            "the market really do worse? Does the quits drop actually come **first** (the whole pitch)? "
            "And if you *sold* every time quits fell, would you make money?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the lead/lag cross-correlation and the "
            "synthetic control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A timing note up front.** JOLTS is published **~6 weeks late** — the March quits rate "
            "isn't out until early May. So an honest trader can only act on a reading **two months after** "
            "the fact. We build that delay in (no peeking). The quits numbers are the official settled "
            "prints (FRED `JTSQUR`), hardcoded so this is reproducible; every chart is drawn by the code "
            "beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When quits fall, does the market do worse? | **Not really.** Over the next few months SPY "
            f"is a touch below average (**+{R['h1'][2]:.1f}%** vs **+{R['h1'][4]:.1f}%** at 1 month) — but "
            "by 6–12 months it's a touch *above*. The sign flips; there's no reliable effect. |\n"
            "| Is any of it reliable? | **No.** Every gap is tiny and well inside the noise — the "
            "strongest reading is a *t* of −0.4, and you can flip the answer just by changing the "
            "lookback. |\n"
            "| Do the quits fall *first*? | **No — this is the killer.** The quits signal lines up best "
            "with a market move that already happened **three-to-four months earlier.** Quits *follow* "
            "stocks; they don't lead them — and then you learn the number 6 weeks late on top. |\n"
            "| So could you trade it? | **It loses.** \"Go to cash when quits fall\" turned $1 into "
            f"**{R['overlay'][8]:.1f}×** vs **{R['overlay'][7]:.1f}×** for just holding — you give up a "
            "third of your money for a Sharpe that's a rounding-tie. |\n\n"
            "> Quits and the business cycle are real. But \"falling quits warn you *early*\" is a "
            "coincident echo wearing a crystal-ball costume — reported too late to act on, and pointing "
            "the wrong way as often as the right one."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The quits rate is worker confidence made visible. People quit only when they're sure "
            "they can do better, so a **falling** quits rate means confidence is cracking — get defensive, "
            "and especially lighten up on cyclicals, because the labour market turns before the stock "
            "market does.\"*\n\n"
            "There's a respectable backbone here. Quits genuinely track the labour cycle — they collapsed "
            "in 2009 (nobody dared quit) and exploded in the 2021 'Great Resignation.' Fed officials watch "
            "the quits rate as a real-time read on labour-market heat. The *trading* leap is the part we "
            "test: that a quits **downturn** arrives early enough, and cleanly enough, to be a "
            "**tradable** warning for equities."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be a gift: a free, monthly, government number telling you to step aside "
            "before drawdowns. But 'the labour market leads' hides a trap. The **stock market is itself a "
            "leading indicator** — it usually turns *before* the economy. So a labour gauge that lines up "
            "with market weakness might not be *predicting* the market at all; it might just be "
            "**echoing** a turn the market already made. Worse, quits are a **slow, sticky** series "
            "reported six weeks late — the opposite of a fast trigger. The difference between *leads* and "
            "*echoes* is the difference between an edge and a mirage, and you can only tell them apart by "
            "checking the timing carefully."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f} years** ({R['start'][:4]}–{R['end'][:4]}, "
            f"{R['months']} months) of the JOLTS quits rate against month-end SPY, and:\n\n"
            "1. **Split the months.** Call quits **falling** when the rate is below where it was three "
            "months ago. Compare what SPY did next (1/3/6/12 months) in falling-quits months vs all "
            "months — entering **two months after** the reference month, because that's when JOLTS is "
            "actually published.\n"
            "2. **Check the timing.** The crucial test: slide quits forward and backward against the "
            "market and find *where* they line up best. If quits truly **lead**, the strongest link shows "
            "up at a **positive lead** (quits first, market later).\n"
            "3. **Try to trade it.** Sit in cash whenever quits are falling, hold otherwise, pay "
            "realistic costs — and see if it beats just buying and holding."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw material.** Here's the quits rate over a quarter-century — the 2009 collapse "
            "(workers too scared to quit), the long climb, and the off-the-chart 2021 'Great Resignation' "
            "spike. Quits clearly *know* about the cycle. The question is whether they know **early**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = F['quits']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(q.index, q.values, c=GREEN, lw=1.4)\n"
            "    ax.set_title('JOLTS quits rate, Total Nonfarm, SA (percent of employment)')\n"
            "    ax.set_ylabel('quits rate (%)')\n"
            "    ax.annotate('2009 trough ~1.2%', xy=(q.idxmin(), q.min()), xytext=(q.idxmin(), 1.5),\n"
            "                arrowprops=dict(arrowstyle='->', color=GREY), color=GREY)\n"
            "    ax.annotate('2021 Great Resignation ~3.0%', xy=(q.idxmax(), q.max()), xytext=(q.idxmax(), 2.6),\n"
            "                ha='right', arrowprops=dict(arrowstyle='->', color=GREY), color=GREY)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('range:', q.min(), 'to', q.max(), '(%)')\n"
            "else:\n"
            "    print('no cache — see docs/results.md; quits ranged 1.2% (2009) to 3.0% (2021)')"
        ),
        md(
            "**Now the payoff.** For each horizon, the average forward SPY return in **falling-quits** "
            "months next to the return on an **average** month. The folklore predicts the red bars sit "
            "*below* the grey ones — at every horizon."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, m) for m in hs]\n"
            "    fal = [r['falling_mean']*100 for r in rows]; base = [r['base_mean']*100 for r in rows]\n"
            "else:\n"
            "    fal = [R['h1'][2], R['h3'][2], R['h6'][2], R['h12'][2]]\n"
            "    base = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if a < b else GREEN for a, b in zip(fal, base)]\n"
            "ax.bar(x-.2, fal, .4, color=cols, label='after quits FALL')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='an average month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('average forward SPY return (%)')\n"
            "ax.set_title('Falling quits -> lower returns? Only barely, and the sign flips by 6 months')\n"
            "for i,(a,b) in enumerate(zip(fal,base)):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('1m: falling', f'{fal[0]:.1f}%', 'vs base', f'{base[0]:.1f}%',\n"
            "      '| 12m: falling', f'{fal[-1]:.1f}%', 'vs base', f'{base[-1]:.1f}%')"
        ),
        md(
            f"Look what happens across the row. At 1–3 months falling-quits returns sit *just* below the "
            f"base rate (the folklore's direction) — but by 6 and 12 months they're *above* it "
            f"(**+{R['h12'][2]:.1f}%** vs **+{R['h12'][4]:.1f}%** at a year). The effect can't decide which "
            "way it points, and every gap is a fraction of a percent. Hold that thought; the *next* chart "
            "is where the story breaks for good."
        ),
        md(
            "**The crucial test: do the quits fall *first*?** We slide quits forward and backward against "
            "the market and measure how tightly they move together. A real leading gauge would show its "
            "strongest link at a **positive lead** (quits lead → bar peaks on the right). Watch where it "
            "actually peaks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F)\n"
            "    Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [GREY if L<0 else GREEN for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=RED, lw=1, ls=':')\n"
            "ax.set_xlabel('lead L (months): L>0 = quits move FIRST (early-warning)   |   L<0 = quits LAG the market')\n"
            "ax.set_ylabel('correlation with market move'); ax.set_xticks(Ls)\n"
            "ax.set_title('The peak is on the LEFT: quits lag the market by ~3-4 months')\n"
            "plt.tight_layout(); plt.show()\n"
            "imax = int(np.nanargmax(cs))\n"
            "print(f'strongest link at L={Ls[imax]} months (quits FOLLOW the market here)')"
        ),
        md(
            f"There it is. The tallest bar is at **L = −4** — the quits signal lines up best with a market "
            "move that happened **three-to-four months earlier**. On the right, where a true early-warning "
            "would live (quits moving first), the bars are near zero. **Quits aren't leading the market — "
            "they're trailing it.** The cycle shows up in stock prices, and only later in the quits rate — "
            "which you then read six weeks after *that*."
        ),
        md(
            "**Could you trade it anyway?** Suppose you sold (went to cash) whenever quits were falling and "
            "held SPY otherwise — acting two months late, as JOLTS forces you to. Here's that strategy's "
            "growth of $1 vs just buying and holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    falling = st.falling_mask(F); pos = (~falling).astype(float).shift(2)\n"
            "    rr = F['spy'].pct_change()\n"
            "    dfp = pd.DataFrame({'r': rr, 'pos': pos}).dropna()\n"
            "    sw = dfp['pos'].diff().abs().fillna(0); c=10/1e4\n"
            "    overlay = (dfp['pos']*dfp['r'] - sw*c)\n"
            "    bh_grow = (1+dfp['r']).cumprod(); ov_grow = (1+overlay).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(bh_grow.index, bh_grow.values, c=GREY, lw=1.8, label='buy & hold SPY')\n"
            "    ax.plot(ov_grow.index, ov_grow.values, c=RED, lw=1.8, label='cash when quits falling (net)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('\"Sell when quits fall\" ends a third poorer than buy-and-hold')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> buy&hold {bh_grow.iloc[-1]:.1f}x  vs  overlay {ov_grow.iloc[-1]:.1f}x')\n"
            "else:\n"
            "    print(f\"overlay {R['overlay'][8]:.1f}x vs buy-hold {R['overlay'][7]:.1f}x terminal — see results.md\")"
        ),
        md(
            f"The defensive overlay ends up **well below** buy-and-hold — "
            f"**{R['overlay'][8]:.1f}×** vs **{R['overlay'][7]:.1f}×** on $1, "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%/yr**. Its Sharpe is a fraction "
            f"higher ({R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}) — but *only* because it's out of the "
            "market ~29% of the time. That's not skill; it's just less exposure. You handed back a third "
            "of your wealth to lower your risk a little — a trade you could make more cheaply by just "
            "holding less SPY."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Falling quits don't reliably precede weaker returns: the effect is "
            "tiny, insignificant (best *t* = −0.4), and it **flips sign** between the short and long "
            "horizon. Point the lookback one way and it's bearish; point it the other and it's bullish. "
            "That's noise.\n"
            "- **Tradability — Mirage.** Selling on falling quits **ends a third poorer** than "
            "buy-and-hold. The whisper of a higher Sharpe is just de-risking — beta you dialled down, not "
            "alpha you found.\n"
            "- **Leading gauge? — Not supported.** The quits drop lines up with a market move that "
            "already happened a quarter earlier — and you read the number six weeks after it's stale. "
            "Quits **echo** equity weakness; they don't forecast it. The one word that makes the pitch — "
            "*early* — is the part the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Forget significance for a second. Even if the tiny tilt were real, the operational reality "
            "kills it twice. First, **JOLTS is six weeks late** — by the time you know March's quits fell, "
            "it's May, and any 'early' warning has long since played out in prices. Second, the claim's "
            "own favourite target — **cyclicals** — actually does *better* after quits fall (cyclicals "
            "lead the recovery the quits reading is lagging into), so the sector bet is backwards. There "
            "is no version of \"sell when quits fall\" that both fires early *and* makes money."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling test.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) "
            "asks the same question of *rising jobless claims* — same result: a coincident labour echo, "
            "not a tradable lead.\n"
            "- **More macro crystal balls.** [Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/) and "
            "[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/) put other celebrated "
            "gauges through the same wringer.\n"
            "- **Build your own.** Swap the quits *rate* for hires, or pair quits with a *price* trend "
            "filter — the lead/lag picture barely budges: a coincident series can't be made to lead by "
            "smoothing it differently, and the publication delay is baked into the data.\n\n"
            "*Think quits lead the market? Show the lead/lag chart peaking on the **right** (positive "
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
            "# JOLTS-Quits — a quantitative teardown 🔬\n"
            "### Quits-momentum split returns · Welch *t* + placebo null · the decisive lead/lag "
            "cross-correlation · a cyclicals leg · a timing overlay vs buy-and-hold · the release-lag "
            "tax · a synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "believers fuse two claims: that a falling quits rate (1) **predicts** weaker equity/cyclical "
            "returns and (2) does so **early** enough to trade. We separate them. The conditional return "
            "tilt is *tiny and sign-unstable*; the decisive object is the **lead/lag structure**, which "
            "shows quits momentum is **coincident-to-lagging**, not leading — and a ~6-week publication "
            "delay plus an overlay that forfeits a third of terminal wealth seals the Tradability axis.\n\n"
            "> ⚠️ **Data + lag note.** The quits tape is a hardcoded monthly snapshot of `JTSQUR` (quits "
            "rate, Total Nonfarm, SA, percent) — the settled print, **not** the real-time vintage (named "
            "on the Signal axis). SPY / XLY / XLP are yfinance daily adjusted close (total-return), "
            "month-end sampled. JOLTS publishes reference month t in month **t+2**, so the execution lag "
            "is **2 months** (no look-ahead), applied once. Offline core + synthetic control are "
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
            f"| **Signal** | `NONE` | Best Welch **t = {R['h1'][7]:.2f}** (1m); the sign **flips** from "
            f"−excess at 1–3m to +excess at 6–12m (12m falling **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%**), down-rate ≈ base everywhere, and a lookback change reverses it. |\n"
            f"| **Tradability** | `MIRAGE` | Cash-on-falling overlay $1→**{R['overlay'][8]:.1f}×** vs "
            f"buy-hold **{R['overlay'][7]:.1f}×**; Sharpe **{R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}** "
            "(a de-risking tie). Beta dialled down, not alpha. |\n"
            f"| **Leading gauge?** | `NOT SUPPORTED` | Peak lead/lag correlation at **L = −4** (quits lag "
            "the market by a quarter); flat at positive leads — then published ~6 weeks late. |\n\n"
            "> 💡 In plain words: the equity market *is* a leading indicator of the economy, so a labour "
            "series that co-moves with equity weakness need not lead it. Quits momentum lines up with a "
            "market move already a quarter old — the 'early-warning' is the market's own lead, reflected "
            "back, and then delayed by the JOLTS release schedule."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $q_t$ be the JOLTS quits rate (percent) and $m_t = q_t - q_{t-3}$ its 3-month change "
            "(percentage points). Quits are **FALLING** at $t$ when $m_t < 0$. With the JOLTS publication "
            "delay the month-$t$ print is public in month $t+2$, so we act at the close of $t+2$ (a "
            "2-month execution lag) and define forward return $r_{t+2\\to t+2+H}$.\n\n"
            "- **H₁ (predicts).** $\\mathbb{E}[r\\mid \\text{falling}] < \\mathbb{E}[r]$ — a *negative* "
            "excess over the base rate, at **every** horizon.\n"
            "- **H₂ (leads).** The strongest positive quits↔return correlation sits at a **positive** "
            "lead (quits move first; falling quits → lower returns ⇒ $\\rho(L{>}0) > 0$).\n"
            "- **H₃ (deployable).** A cash-on-falling overlay beats buy-and-hold net of costs.\n\n"
            "We find **H₁ rejected** (tiny, insignificant, and *sign-unstable* across horizon), "
            "**H₂ rejected** (peak positive corr at $L=-4$), **H₃ rejected** (overlay forfeits a third of "
            "terminal wealth). The folklore is right only where it's uninformative (quits and the cycle "
            "co-move) and wrong exactly where it would pay (a *leading*, *tradable* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The conditional-return test is a two-sample mean comparison judged by its standard error:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{falling}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{falling}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "But a significant $\\widehat{\\Delta}$ would **still not** establish *leading*: a coincident "
            "or lagging series can co-move with forward returns through cycle autocorrelation. The "
            "identifying test is the **lead/lag cross-correlation** "
            "$\\rho(L) = \\mathrm{corr}(m_t,\\ r_{t+L\\to t+L+1})$. A genuine early-warning peaks "
            "(positively) at $L>0$. If $\\arg\\max_L \\rho(L) < 0$, quits **follow** the market — and the "
            "entire 'early' thesis collapses regardless of the conditional mean. On top of that sits an "
            "*institutional* constraint the other macro studies don't face: the **2-month JOLTS "
            "publication delay**, which taxes any residual lead before you can act."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Quits tape.** Monthly `JTSQUR` (percent, SA), hardcoded snapshot, "
            f"{R['start'][:7]}→{R['end'][:7]} ({R['months']} months). Settled print, not real-time "
            "vintage (named on the axis).\n"
            "- **Signal.** $m_t = q_t-q_{t-3}$; FALLING when $m_t<0$.\n"
            "- **Forward returns.** Enter at the close **2 months after** the reference month (the JOLTS "
            "release lag — no look-ahead), hold $H\\in\\{1,3,6,12\\}$ months; drop horizons that overrun "
            "the tape. Run on **SPY** and on a **cyclical-minus-defensive** (XLY−XLP) long-short.\n"
            "- **Null #1 (Welch t).** Falling-set mean vs the unconditional mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random months; "
            "$p = \\Pr[\\text{random-draw mean} \\le \\text{falling mean}]$ (as bearish or more).\n"
            "- **Identification (lead/lag).** $\\rho(L)$ for $L\\in[-6,6]$ — *where* do quits line up?\n"
            "- **Tradability.** Cash-when-falling overlay, 2-month lag, 10 bps one-way per switch "
            "(turnover one-way × NAV), excess-of-zero Sharpe (cash leg = 0, labelled).\n"
            "- **Positive control.** A deterministic series with a *planted* quits→returns link: "
            "`edge=0` must not fake bearish significance; a large `edge` must light up the test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — tiny, insignificant, and sign-unstable\n\n"
            "Falling-quits forward mean with $\\pm$ standard error against the unconditional base rate "
            "(dashed). Below base at 1–3m, **above** base at 6–12m — every gap inside its own error bar."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts, ses = [], [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize(F, m); cm.append(s['falling_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        d,_u,_a = st.split_returns(F, m); ses.append(d.std(ddof=1)/np.sqrt(len(d)))\n"
            "else:\n"
            "    cm = [R['h1'][2]/100, R['h3'][2]/100, R['h6'][2]/100, R['h12'][2]/100]\n"
            "    bm = [R['h1'][4]/100, R['h3'][4]/100, R['h6'][4]/100, R['h12'][4]/100]\n"
            "    ts = [R['h1'][7], R['h3'][7], R['h6'][7], R['h12'][7]]; ses = [.012,.02,.03,.045]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [RED if c<b else GREEN for c,b in zip(cm,bm)]\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=cols, width=.5, label='falling-quits (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Sign flips with horizon; the SE swamps every gap'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 1m the falling-quits mean is **+{R['h1'][2]:.1f}%** vs base "
            f"**+{R['h1'][4]:.1f}%** (right sign, **t = {R['h1'][7]:.2f}**); at 12m it's "
            f"**+{R['h12'][2]:.1f}%** vs **+{R['h12'][4]:.1f}%** — *above* base, **t = {R['h12'][7]:+.2f}**. "
            "H₁ is **rejected**: not just insignificant but *directionally incoherent*. A real gauge "
            "doesn't change its mind about the sign between quarter one and year one."
        ),
        md(
            "### 4b · The decisive identification test — lead/lag\n\n"
            "$\\rho(L) = \\mathrm{corr}(m_t, r_{t+L\\to t+L+1})$. Positive bars left of zero = quits "
            "**lag** the market; a real leading gauge would peak on the **right** (quits lead)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [GREY if L<0 else GREEN for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=RED, lw=1, ls=':')\n"
            "imax = int(np.nanargmax(cs))\n"
            "ax.annotate('strongest link\\n(quits LAG the market)', xy=(Ls[imax], cs[imax]),\n"
            "            xytext=(Ls[imax]-0.3, cs[imax]+0.06), ha='center', color=GREY,\n"
            "            arrowprops=dict(arrowstyle='->', color=GREY))\n"
            "ax.set_xlabel('lead L (months): L>0 = quits lead (early-warning)   |   L<0 = quits lag')\n"
            "ax.set_ylabel(r'$\\rho(L)$'); ax.set_xticks(Ls)\n"
            "ax.set_title('argmax rho(L) is at L<0: quits are coincident-to-lagging')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'argmax at L={Ls[imax]} (rho={cs[imax]:+.2f}); rho at +1 month = {cs[Ls.index(1)]:+.2f}')"
        ),
        md(
            "> 💡 In plain words: $\\arg\\max_L \\rho(L) = -4$. The quits signal correlates most with a "
            "market move **a quarter in its past**; at the positive leads a genuine early-warning needs, "
            "$\\rho \\approx +0.05$. **H₂ rejected.** The equity market leads the economy; quits trail "
            "both — the 'early-warning' is the market's lead, reflected. This is the load-bearing result, "
            "independent of the conditional-mean significance, and it survives before the release lag is "
            "even charged."
        ),
        md(
            "### 4c · The cyclicals leg — the claim's own target goes the wrong way\n\n"
            "The believers single out cyclicals. So we run the split on a monthly-rebalanced **XLY − XLP** "
            "(discretionary minus staples) long-short — the risk-appetite tape. Falling quits should make "
            "it *negative*; instead it's *more positive* than base."
        ),
        code(
            "if HAVE_REAL and 'cyc' in F.columns:\n"
            "    hs2 = [1,3,6,12]; cf=[]; cb=[]; ct=[]\n"
            "    for m in hs2:\n"
            "        s = st.summarize(F, m, price='cyc'); cf.append(s['falling_mean']*100); cb.append(s['base_mean']*100); ct.append(s['t'])\n"
            "else:\n"
            "    hs2 = [1,3,6,12]; cf=[c[2] for c in R['cyc']]; cb=[c[3] for c in R['cyc']]; ct=[c[4] for c in R['cyc']]\n"
            "x = np.arange(len(hs2))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cf, .4, color=GREEN, label='cyclicals-minus-defensives after quits FALL')\n"
            "ax.bar(x+.2, cb, .4, color=GREY, label='base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs2]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('forward XLY-XLP return (%)'); ax.legend()\n"
            "ax.set_title('Falling quits -> cyclicals OUTPERFORM (wrong sign for the claim)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cyc 12m: falling', f'{cf[-1]:.1f}%', 'vs base', f'{cb[-1]:.1f}%', 't=', round(ct[-1],2))"
        ),
        md(
            f"> 💡 In plain words: at 12m the cyclical spread is **+{R['cyc'][3][2]:.1f}%** after falling "
            f"quits vs **+{R['cyc'][3][3]:.1f}%** base (**t = {R['cyc'][3][4]:+.2f}**) — the *opposite* of "
            "the claim, and still insignificant. Cyclicals lead the recovery that the lagging quits "
            "reading is sinking into, so 'sell cyclicals when quits fall' is, if anything, a fade of the "
            "next up-leg."
        ),
        md(
            "### 4d · Tradability — the cash-on-falling overlay forfeits a third of the pot\n\n"
            "Hold SPY when quits rise, cash when falling (2-month lag, 10 bps/switch). Annualised mean, "
            "Sharpe, and terminal wealth vs buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    o = st.timing_overlay(F, cost_bps=10.0)\n"
            "    bh_m, bh_s = o['bh_mean']*100, o['bh_sharpe']\n"
            "    g_m, g_s = o['overlay_gross_mean']*100, o['overlay_gross_sharpe']\n"
            "    n_m, n_s = o['overlay_net_mean']*100, o['overlay_net_sharpe']; nsw=o['n_switches']\n"
            "    falling = st.falling_mask(F); pos=(~falling).astype(float).shift(2); rr=F['spy'].pct_change()\n"
            "    dfp = pd.DataFrame({'r':rr,'pos':pos}).dropna(); sw=dfp['pos'].diff().abs().fillna(0)\n"
            "    ov=(dfp['pos']*dfp['r']-sw*10/1e4); bh_T=(1+dfp['r']).cumprod().iloc[-1]; ov_T=(1+ov).cumprod().iloc[-1]\n"
            "else:\n"
            "    bh_m, bh_s = R['overlay'][0], R['overlay'][1]; g_m,g_s = R['overlay'][2],R['overlay'][3]\n"
            "    n_m, n_s = R['overlay'][4], R['overlay'][5]; nsw=R['overlay'][6]; bh_T=R['overlay'][7]; ov_T=R['overlay'][8]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "labels = ['buy &\\nhold', 'overlay\\ngross', 'overlay\\nnet @10bps']\n"
            "a1.bar(labels, [bh_s, g_s, n_s], color=[GREY, AMBER, GREEN], width=.6)\n"
            "for i,v in enumerate([bh_s,g_s,n_s]): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('annualised Sharpe (excess-of-0)'); a1.set_title(f'Sharpe: a de-risking tie ({nsw} switches)')\n"
            "a2.bar(['buy & hold','overlay net'], [bh_T, ov_T], color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([bh_T,ov_T]): a2.annotate(f'{v:.1f}x',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('terminal $1 (total-return)'); a2.set_title('Terminal wealth: overlay gives back a third')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net overlay {n_m:.1f}%/yr (Sharpe {n_s:.2f}, {ov_T:.1f}x) vs buy-hold {bh_m:.1f}%/yr (Sharpe {bh_s:.2f}, {bh_T:.1f}x)')"
        ),
        md(
            f"> 💡 In plain words: the overlay's Sharpe (**{R['overlay'][5]:.2f}**) barely tops "
            f"buy-and-hold's (**{R['overlay'][1]:.2f}**) — a rounding-tie — and buys it only by sitting in "
            f"cash ~{R['frac_falling']}% of months, forfeiting **~2 pts/yr** and ending at "
            f"**{R['overlay'][8]:.1f}× vs {R['overlay'][7]:.1f}×**. That Sharpe wobble is **de-risking**, "
            "reproducible by just holding less SPY at zero research cost. **H₃ rejected.** `MIRAGE`."
        ),
        md(
            "### 4e · Robustness + the publication-delay tax\n\n"
            "Vary the momentum window $k$, the threshold, and drop 2020–2021 — then check the execution "
            "lag. The 12-month *t* is a coin-flip of the window, and even the (untradable) look-ahead lag "
            "is near zero, so there's no edge for the delay to erode."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for k in (1,3,6):\n"
            "        s = st.summarize(F, 12, k=k); rob.append((f'k={k}', s['n_falling'], s['t']))\n"
            "    s = st.summarize(F, 12, thresh=0.2); rob.append(('thr>0.2pp', s['n_falling'], s['t']))\n"
            "    F2 = F[(F.index < '2020-01-01') | (F.index >= '2022-01-01')]\n"
            "    s = st.summarize(F2, 12); rob.append(('ex-COVID', s['n_falling'], s['t']))\n"
            "    lags = {lg: st.summarize(F, 12, lag=lg)['t'] for lg in (0,1,2)}\n"
            "else:\n"
            "    rob = [(r[0], r[1], r[4]) for r in R['robust']]; lags = R['lag']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "labels = [r[0] for r in rob]; tt = [r[2] for r in rob]; nn = [r[1] for r in rob]\n"
            "cols = [RED if t<0 else GREEN for t in tt]\n"
            "a1.bar(labels, tt, color=cols, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "a1.axhline(-2, ls='--', c=RED, label='|t|=2 bar'); a1.axhline(2, ls='--', c=RED)\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): a1.annotate(f'n={k}',(i,t),ha='center',va='bottom' if t>=0 else 'top', fontsize=8)\n"
            "a1.set_ylabel('Welch t (12m)'); a1.set_title('Sign flips with the window; nothing near |t|=2'); a1.legend(fontsize=8)\n"
            "a1.tick_params(axis='x', labelrotation=20)\n"
            "lk = sorted(lags); a2.bar([f'{l}m' for l in lk], [lags[l] for l in lk], color=[GREY,AMBER,GREEN], width=.6)\n"
            "for i,l in enumerate(lk): a2.annotate(f'{lags[l]:+.2f}',(i,lags[l]),ha='center',va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('Welch t (12m)')\n"
            "a2.set_title('Even the look-ahead (0m) lag is ~0: no lead to erode')\n"
            "a2.set_xlabel('execution lag (2m = honest JOLTS release)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label,n,t):', [(r[0], r[1], round(r[2],2)) for r in rob])\n"
            "print('lag sensitivity (t):', {k: round(v,2) for k,v in sorted(lags.items())})"
        ),
        md(
            "> 💡 In plain words: the slow (k=6) filter turns *positive* "
            f"(**t={R['robust'][2][4]:+.2f}**), the fast (k=1) filter *negative* "
            f"(**t={R['robust'][0][4]:+.2f}**), ex-COVID is **t={R['robust'][4][4]:+.2f}** — a sign you can "
            "choose. And the *look-ahead* 0-month lag is only "
            f"**t={R['lag'][0]:+.2f}**, so there was never a timing edge for the 6-week delay to tax. Real "
            "only where you cherry-pick, useless everywhere it would pay."
        ),
        md(
            "### 4f · Faithful-engine control — we know the truth here\n\n"
            "A deterministic monthly series with a *planted* link (falling quits momentum at $t$ depresses "
            "the $t{+}1$ return by `edge`). With `edge=0` the bearish test must stay flat; with a large "
            "`edge` it must light up — proving the engine is unbiased and the real-tape null isn't a "
            "measurement failure."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.04):\n"
            "    syn = data.synthetic_quits(n_months=300, edge=edge, seed=755)\n"
            "    s = st.summarize(syn, 1, k=3, lag=1)\n"
            "    res.append((edge, s['n_falling'], s['falling_mean']*100, s['base_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / month' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t=-2 (bearish significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('Welch t (1-month)'); ax.set_title('Control: no link -> no bearish false positive; real link -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e*100:+.0f}%/mo: n_fall={k} falling={c:.2f}% base={b:.2f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted link the bearish test shows no false positive "
            f"(**t = {R['syn'][0][4]:+.2f}**, placebo **p = {R['syn'][0][5]:.2f}** — the opposite of "
            f"significant-and-negative); a **+4%/month** planted link drives **t = {R['syn'][1][4]:.2f}** "
            f"(**p = {R['syn'][1][5]:.3f}**). So the machinery is honest — the real-tape near-zero *t* is a "
            "*genuine* absent edge, not a broken test. The engine *can* bank a real quits→returns link; "
            "the real tape just doesn't carry one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — best Welch **t = {R['h1'][7]:.2f}**; the excess **flips sign** from "
            f"−0.2pp at 1m to **+{R['h12'][2]-R['h12'][4]:+.1f}pp** at 12m, the down-rate matches base at "
            "every horizon, and the window choice reverses the sign. The named cyclicals leg goes the "
            "**wrong** way. Not a certified signal — noise.\n"
            f"- **Tradability `MIRAGE`** — the cash-on-falling overlay ends at "
            f"**{R['overlay'][8]:.1f}×** vs buy-hold **{R['overlay'][7]:.1f}×** on $1 "
            f"(**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%/yr**); the Sharpe "
            f"**{R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}** tie is de-risking, not alpha. Nothing to "
            "allocate to.\n"
            "- **Leading gauge? `NOT SUPPORTED`** — $\\arg\\max_L \\rho(L) = -4$ months: quits momentum is "
            "**coincident-to-lagging**, not leading; flat at positive leads, and then published ~6 weeks "
            "late. The equity market is the leading indicator; quits echo it. The defining word — "
            "*early* — is the part the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even a real tilt wouldn't deploy\n\n"
            "Grant the lore a genuine few-basis-point tilt. Two structural facts defeat it. **(1) The "
            "release lag.** JOLTS reports reference month t in month t+2, so the *fastest* honest action "
            "is already ~6 weeks stale — and since even the look-ahead (0-lag) test is ≈0, there is no "
            "lead to be late for. **(2) The wrong target.** The overlay is out of the market ~29% of the "
            "time, forfeiting a third of terminal wealth for a Sharpe wobble, while the claim's favourite "
            "trade — short cyclicals on falling quits — has the **wrong sign** (cyclicals lead the "
            "recovery the quits reading is lagging into). No lag, threshold, or cost assumption rescues a "
            "coincident series, published late, masquerading as a leading one."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sibling.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/): "
            "*rising* claims as an early-warning, same hardcoded-snapshot + SPY method — another "
            "coincident labour echo, not a lead.\n"
            "- **Companion macro nowcasts.** [Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/), "
            "[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/) — does any celebrated "
            "macro gauge time equities?\n"
            "- **Sharper identification.** Use **real-time vintages** (ALFRED) to kill any revision "
            "look-ahead, or run a proper VAR / Granger test on the weekly-equivalent series; the "
            "coincident-to-lagging structure and the publication delay are robust to all of these — "
            "smoothing a coincident series differently can't manufacture a lead.\n\n"
            "*The reproducible core is offline and deterministic; the quits input is an explicit frozen "
            "snapshot. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
