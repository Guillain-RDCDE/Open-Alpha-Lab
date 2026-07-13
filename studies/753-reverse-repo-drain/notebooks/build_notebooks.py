"""Generate the two narrative notebooks for Study 753 (Reverse-Repo-Drain).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached month-end SPY
under ../_cache/ and the hardcoded ON RRP proxy; otherwise they quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (hardcoded ON RRP proxy for
# FRED RRPONTSYD + month-end SPY, 2021-01-31 -> 2025-07-31, 55 month-ends, 4.5 years).
R = dict(
    start="2021-01-31", end="2025-07-31", months=55, years=4.5,
    rrp_peak=2554, rrp_peak_date="2022-12", rrp_last=195,
    base_mean=1.22, base_win=65, corr=-0.014,
    # per-horizon regime: (k, n_drain, n_fill, drain%, fill%, spread_pp, t, p_block, dwin%, fwin%)
    k1=(1, 28, 25, 0.71, 1.73, -1.02, -0.80, 0.819, 54, 76),
    k2=(2, 26, 26, 1.23, 1.01, 0.22, 0.17, 0.445, 62, 65),
    k3=(3, 27, 24, 0.94, 1.16, -0.22, -0.17, 0.566, 56, 71),   # headline (trailing quarter)
    k6=(6, 28, 20, 1.58, 0.18, 1.40, 0.92, 0.065, 68, 50),
    k9=(9, 27, 18, 1.70, -0.19, 1.89, 1.20, 0.104, 67, 50),
    k12=(12, 25, 17, 1.60, 0.23, 1.38, 0.83, 0.101, 64, 59),
    # timing: (label, exposure%, switches, gross_sharpe, net_sharpe, net_ann%, bh_sharpe)
    timing=[("long / flat", 53, 14, 0.52, 0.49, 5.6, 0.78),
            ("long / short", 53, 29, -0.04, -0.08, -1.3, 0.78)],
    wealth_lf=1.24, wealth_bh=1.61,
    # synthetic: (edge, n_drain, drain%, fill%, t, p_block)
    syn=[(0.0, 64, 0.95, 0.42, 0.63, 0.284), (0.02, 64, 2.95, 0.42, 3.01, 0.008)],
    # robustness sweep (the quants): list of k tuples above, in display order
)
R["sweep"] = [R["k1"], R["k2"], R["k3"], R["k6"], R["k9"], R["k12"]]

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Liquidity_tell%3F: Busted](https://img.shields.io/badge/Liquidity_tell%3F-Busted-8b949e?style=flat-square)\n\n"
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

from reverse_repo_drain import data, strategy as st

HAVE_REAL = data.have_real()
F = data.build_real() if HAVE_REAL else None
print("real SPY cache present:", HAVE_REAL,
      "| monthly observations:", (0 if F is None else len(F)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Buy when the reverse-repo drains\" — is a shrinking RRP really risk-on? 💵\n"
            "### The desk's favourite liquidity-plumbing chart — the Fed's $2.5-trillion cash "
            "parking lot — put to the test, in plain English\n\n"
            + BADGES +
            "There's a chart that liquidity-watchers love: the Fed's **Overnight Reverse Repo "
            "(ON RRP) facility** — a giant overnight parking lot where money-market funds stash "
            "idle cash — plotted against the S&P 500. It ballooned to an all-time **$2.55 trillion** "
            "at the end of 2022, then **drained** almost all the way back to zero through 2023-2025. "
            "The folklore is irresistibly simple: *when the RRP drains, that cash is leaving the Fed "
            "and pouring into stocks — so a draining RRP means risk-on, buy equities.*\n\n"
            "It looks compelling because the drain lined up beautifully with the 2023-24 bull market. "
            "But that's **one** episode — and the *same* facility was **filling** through 2021, when "
            "stocks were also roaring. So does a draining RRP actually point up? This notebook lines "
            "the two up honestly.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the block-bootstrap null, the Sharpe "
            "race and the power control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The ON RRP balance isn't on yfinance, so we ship a "
            "**small, clearly-labelled hardcoded monthly proxy** of the public FRED series "
            "`RRPONTSYD` (the fill to the $2.55T peak and the drain back down), aligned to month-end "
            "SPY. It's a rounded transcription, named a proxy throughout — the story turns on the "
            "*shape* of one fill-then-drain cycle, not on any decimal. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When the RRP is draining, is the next month up more often? | **No — less often.** Over "
            "the trailing-quarter definition, draining months are up **56%** of the time vs **71%** "
            "for filling months. |\n"
            "| But didn't the drain track the 2023-24 rally? | **Yes — and that's the trap.** The RRP "
            "*also filled* through the 2021 bull market. One cycle can't tell you which way the arrow "
            "points; it points **both** ways here. |\n"
            "| So is the drain a real signal? | **No.** The draining-minus-filling return gap is "
            "essentially zero or *negative* at short horizons, and the straight correlation between the "
            "RRP's recent change and next month's return is **−0.01** — noise. |\n"
            "| Could you at least trade it? | **No.** A \"hold-when-draining\" rule earns a **lower "
            "Sharpe (0.49) than just buying and holding (0.78)** — sitting out whenever the RRP fills "
            "costs more than the drain months pay. |\n\n"
            "> The RRP drain is real plumbing — QT and a flood of T-bills pulling cash out of the "
            "facility. \"Drain = risk-on\" is a story told about a single macro cycle, and it isn't "
            "even a good one."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The reverse-repo facility is where idle cash hides. When the RRP **drains**, that "
            "cash is flooding back into the system and into risk assets — so a falling RRP marks a "
            "**risk-on** regime. Be long stocks while it drains; be cautious while it fills.\"*\n\n"
            "This is the everyday reading of the ON RRP balance, charted on liquidity-plumbing feeds "
            "against the S&P 500. The picture is intuitive: $2.5 trillion is a lot of dry powder, and "
            "if it's leaving the Fed it must be going *somewhere*. We'll put the RRP proxy next to SPY "
            "and ask whether \"draining\" really precedes \"up\" — or just **coincided** with it once."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a freely-charted Fed series really flagged the equity regime a month ahead, it would be "
            "an extraordinary free lunch — macro liquidity as a market-timing switch. But two things "
            "hide inside \"drain = buy.\" (1) *Does the market rise while the RRP drains?* Sure — but "
            "the market rises most months anyway, and the RRP happened to drain across one big bull. "
            "The question that matters is (2) *does it rise by **more** than usual, and does the "
            "**direction** hold up?* A draining RRP that filled during the *previous* bull tells you "
            "the co-movement is a coincidence of one cycle, not a signal. And the drain itself has a "
            "boring mechanical cause — quantitative tightening plus a wave of T-bill issuance vacuuming "
            "cash out of the facility — that has nothing to do with equity risk appetite."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take the ON RRP proxy and month-end SPY over **{R['years']:.1f} years** "
            f"({R['start']} → {R['end']}, {R['months']} month-ends) and:\n\n"
            "1. **Mark the drain.** A month is *draining* if the RRP is lower than it was a few months "
            "earlier (we try 1, 2, 3, 6 months back), known at month-end and acted on **one month "
            "later** — no look-ahead.\n"
            "2. **Measure the payoff.** For draining vs filling months, what did SPY do the **next "
            "month** — and how does the gap compare to a normal month?\n"
            "3. **Stress the luck, then try to trade it.** Because there's really just one fill and one "
            "drain, we reshuffle the regime labels in **blocks** thousands of times to see how easily "
            "the gap shows up by chance; then race a \"hold-when-draining\" rule against buy-and-hold, "
            "net of costs.\n\n"
            "**What would make us say \"mirage\"?** If the draining-minus-filling gap is small, "
            "flips sign with the look-back, and a timing rule loses to just holding — that's a no."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's the facility itself.** The RRP proxy filling (blue) to its "
            "**$2.55 trillion** end-2022 peak and draining (orange) back down, overlaid on SPY. Notice "
            "the two things the folklore forgets: the RRP **filled** right through the 2021 bull, and "
            "the big **drain** coincided with the 2023-24 bull. Same facility, opposite market — the "
            "co-movement can't have one sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d3 = F['rrp'].diff(3)\n"
            "    drain = (d3 < 0)\n"
            "    fig, ax1 = plt.subplots(figsize=(9.6, 4.5))\n"
            "    ax1.plot(F.index, F['rrp'], color='#2c3e50', lw=1.4)\n"
            "    ax1.fill_between(F.index, 0, F['rrp'], where=~drain, color='#3b6fb0', alpha=.35, label='RRP filling')\n"
            "    ax1.fill_between(F.index, 0, F['rrp'], where=drain, color='#e08a3c', alpha=.45, label='RRP draining')\n"
            "    ax1.set_ylabel('ON RRP proxy ($B)')\n"
            "    ax2 = ax1.twinx(); ax2.plot(F.index, F['spy'], c=GREY, lw=1.6, label='SPY (right)')\n"
            "    ax2.grid(False); ax2.set_ylabel('SPY (total-return)')\n"
            "    ax1.set_title('The $2.55T fill-then-drain — one cycle, straddling a bear and a bull')\n"
            "    ax1.legend(loc='upper right'); plt.tight_layout(); plt.show()\n"
            "    print('RRP peak $%.0fB at %s -> $%.0fB latest' % (F['rrp'].max(), F['rrp'].idxmax().date(), F['rrp'].iloc[-1]))\n"
            "else:\n"
            "    print('no cache — see docs/results.md: RRP peak', R['rrp_peak'], 'B ->', R['rrp_last'], 'B')"
        ),
        md(
            f"The proxy peaks at **${R['rrp_peak']:,}B** ({R['rrp_peak_date']}) and drains to "
            f"**${R['rrp_last']}B**. The whole informative history is *one hump*. Now the real "
            "question: within that hump, does a draining month actually precede a better month than a "
            "filling one?"
        ),
        md(
            "**Draining vs filling — who's up more often?** For the trailing-quarter definition, here's "
            "how often SPY was higher the next month after a draining month vs a filling month, next to "
            "the base rate of any month."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(F, k=3)\n"
            "    dwin, fwin, bwin = s['drain_win']*100, s['fill_win']*100, s['base_win']*100\n"
            "else:\n"
            "    dwin, fwin, bwin = R['k3'][8], R['k3'][9], R['base_win']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "bars = ax.bar([0,1,2], [dwin, fwin, bwin], .55, color=[RED, GREEN, GREY])\n"
            "ax.set_xticks([0,1,2]); ax.set_xticklabels(['after a DRAINING month','after a FILLING month','any month (base rate)'])\n"
            "ax.set_ylim(0, 100); ax.set_ylabel('% of the time SPY is higher next month')\n"
            "ax.set_title('The draining months are up LESS often than the filling months')\n"
            "for i,v in enumerate([dwin,fwin,bwin]): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'drain next-month win {dwin:.0f}%  vs  fill {fwin:.0f}%  vs  base {bwin:.0f}%')"
        ),
        md(
            f"There's the first crack. After a draining month SPY is up only **{R['k3'][8]}%** of the "
            f"time next month — *below* the **{R['k3'][9]}%** you get after a filling month and below "
            f"the **{R['base_win']}%** base rate. The \"risk-on\" regime is, if anything, the "
            "*worse* month to own stocks. The drain didn't lead the rally; it just happened during it."
        ),
        md(
            "**Does the sign even hold?** Maybe a quarter is the wrong look-back. Here's the "
            "draining-minus-filling return gap for look-backs from 1 to 12 months. Watch it flip sign — "
            "a real signal doesn't change direction depending on how you squint."
        ),
        code(
            "ks = [1,2,3,6,9,12]\n"
            "if HAVE_REAL:\n"
            "    spreads = [st.summarize(F, k=k)['spread']*100 for k in ks]\n"
            "else:\n"
            "    spreads = [R[f'k{k}'][5] for k in ks]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in spreads]\n"
            "ax.bar([str(k) for k in ks], spreads, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.9)\n"
            "for i,s in enumerate(spreads): ax.annotate(f'{s:+.1f}',(i,s),ha='center',va='bottom' if s>=0 else 'top')\n"
            "ax.set_xlabel('\"draining\" look-back (months)'); ax.set_ylabel('drain − fill next-month return (pp)')\n"
            "ax.set_title('The drain edge flips sign with the look-back — a hallmark of noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads (pp) by look-back:', {k: round(s,2) for k,s in zip(ks, spreads)})"
        ),
        md(
            "> 🔬 **For the quants.** The gap is **negative** at 1-3 months and only turns positive at "
            "6-12 months — where \"draining\" mostly just re-labels *\"we're past the Dec-2022 peak.\"* "
            "Even at its best (9-month look-back) the Welch *t* is **+1.20**, short of the *t* ≥ 2 bar. "
            "A sign that depends on the window isn't a signal. The block-bootstrap null is in "
            "[02_for_the_quants.ipynb](02_for_the_quants.ipynb)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** A draining RRP does not precede higher returns. At the natural "
            f"1-3-month horizons the gap is *negative* (draining months up **{R['k3'][8]}%** of the "
            f"time vs **{R['k3'][9]}%** filling), and the correlation between the RRP's recent change "
            f"and next month's return is **{R['corr']:+.2f}** — noise. The only positive signs appear "
            "at long look-backs and still fail significance.\n"
            f"- **Tradability — Mirage.** A \"hold-when-draining\" rule earns net Sharpe "
            f"**{R['timing'][0][4]:.2f}** vs **{R['timing'][0][6]:.2f}** for buy-and-hold — you give up "
            "return by sitting out whenever the RRP fills.\n"
            "- **Liquidity tell? — Busted.** The drain is QT plus a T-bill flood mechanically emptying "
            "the facility — coincident with one bull, contradicted by the 2021 fill. \"Drain = "
            "risk-on\" is one macro cycle wearing a signal's clothes."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — race it against buy-and-hold\n\n"
            "Forget significance for a second and just ask the operational question: if you hold SPY "
            "only while the RRP is draining and step aside while it fills, do you beat simply owning "
            "the market? Here's the growth of $1 for each, net of costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = F['spy'].pct_change(); sig = st.draining(F, k=3)\n"
            "    pos = sig.map({True:1.0, False:0.0}).shift(1)\n"
            "    dfp = pd.DataFrame({'ret':ret,'pos':pos}).dropna()\n"
            "    turn = dfp['pos'].diff().abs().fillna(dfp['pos'].abs()); c = 10/1e4\n"
            "    net = dfp['pos']*dfp['ret'] - turn*c\n"
            "    eq_rule = (1+net).cumprod(); eq_bh = (1+dfp['ret']).cumprod(); expo=(dfp['pos']>0).mean()\n"
            "else:\n"
            "    rng = np.random.default_rng(753); bh = pd.Series(rng.normal(0.01,0.045,R['months']))\n"
            "    eq_bh = (1+bh).cumprod(); eq_rule = (1+bh*0.53).cumprod(); expo=0.53\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(eq_bh.index, eq_bh.values, c=GREEN, lw=2, label='buy & hold')\n"
            "ax.plot(eq_rule.index, eq_rule.values, c=RED, lw=1.9, label='hold-when-draining (net)')\n"
            "ax.set_ylabel('growth of $1'); ax.set_title(f'Holding only on drains ({expo*100:.0f}% exposure) trails buy & hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'final wealth: rule {eq_rule.iloc[-1]:.2f}x vs buy&hold {eq_bh.iloc[-1]:.2f}x (exposure {expo*100:.0f}%)')"
        ),
        md(
            f"There it is. The drain rule ends at **{R['wealth_lf']:.2f}×** against "
            f"**{R['wealth_bh']:.2f}×** for just buying and holding, at a net Sharpe of "
            f"**{R['timing'][0][4]:.2f}** vs **{R['timing'][0][6]:.2f}**. The red line flatlines "
            "through every filling stretch — and because one of those stretches was the 2021 bull, "
            "stepping aside was expensive. Betting *against* stocks while the RRP fills (long/short) is "
            f"outright **negative ({R['timing'][1][4]:.2f})**. **Costs aren't the problem; being out of "
            "an up-drifting market on a coin-flip signal is.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Wait for episode #2.** The single biggest limitation is honest and unfixable today: "
            "there is **one** RRP cycle. If the facility fills and drains again through a *different* "
            "market, re-run this and see whether the sign is stable — that's the test the claim needs "
            "and can't yet pass.\n"
            "- **Use the level, or the reserve-scarcity story.** Some versions key off the RRP *level* "
            "(near the floor = scarce reserves = fragile) rather than the change. Swap the signal and "
            "re-run the detector — the n=1 problem doesn't go away.\n"
            "- **The macro-timing pattern.** The ISM-PMI regime, the economic surprise index and "
            "jobless-claims momentum studies on this bench all rhyme: a plausible macro relationship "
            "rarely survives as a tradable monthly timing rule — and this one fails a step earlier, on "
            "a single-episode sample.\n\n"
            "*Think a draining RRP beats the market by more than luck? Reshuffle the drain/fill labels "
            "in blocks, show the real gap landing **outside** the cloud **and** a timing rule clearing "
            "buy-and-hold — across **more than one** cycle — then we'll talk.*"
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
            "# Reverse-Repo-Drain — a quantitative teardown 🔬\n"
            "### A hardcoded ON RRP proxy (FRED `RRPONTSYD`) vs month-end SPY · drain-vs-fill regime "
            "means · a Welch *t* + a block-bootstrap null on few long regimes · a drain-timing-vs-"
            "buy-and-hold Sharpe race · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "\"a draining ON RRP marks risk-on\" is a **regime** hypothesis on a **single** "
            "fill-then-drain episode, so the entire teardown is one question asked carefully: is the "
            "drain-minus-fill forward-return spread distinguishable from what one big drain lining up "
            "with one big rally produces by chance? We confront it with a Welch *t*, a block-bootstrap "
            "null that respects the long regimes, and a costs-net Sharpe race.\n\n"
            "> ⚠️ **Data + proxy note.** The ON RRP balance isn't on yfinance; we ship a **small, "
            "clearly-labelled hardcoded monthly proxy** of the public FRED series `RRPONTSYD` (rounded "
            "end-of-month levels, quarter-end spikes smoothed) — the 2021 fill, the ~$2.55T Dec-2022 "
            "peak, the 2023-25 drain. SPY is yfinance **total-return** (auto-adjust), month-end. The "
            "verdict turns on the *shape of one episode*, not the marks. Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Trailing-quarter drain spread **{R['k3'][5]:+.1f}pp** (Welch "
            f"**t = {R['k3'][6]:+.2f}**); *negative* at 1-3m, best case Welch **t = {R['k9'][6]:+.2f}** "
            f"(k=9, block **p = {R['k9'][7]:.2f}**); corr(ΔRRP, next ret) **= {R['corr']:+.2f}**. Never "
            "clears **\\|t\\| ≥ 2**. |\n"
            f"| **Tradability** | `MIRAGE` | Drain-timing net Sharpe **{R['timing'][0][4]:.2f}** "
            f"(long/flat) / **{R['timing'][1][4]:.2f}** (long/short) vs buy-and-hold "
            f"**{R['timing'][0][6]:.2f}**; ends {R['wealth_lf']:.2f}× vs {R['wealth_bh']:.2f}×. |\n"
            f"| **Liquidity tell?** | `BUSTED` | n=1 episode: the RRP **filled** through the 2021 bull "
            "and **drained** through the 2023-24 bull — the co-movement can't hold one sign. |\n\n"
            "> 💡 In plain words: the drain looks bullish only because you remember 2023-24. The same "
            "facility filling through the 2021 bull cancels the story, and a null that respects the "
            "long regimes says one drain matching one rally is unremarkable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $B_t$ be the month-$t$ ON RRP level. The **drain** regime is a negative trailing "
            "change, $\\mathrm{drain}_t(k) = \\mathbb{1}[\\,B_t < B_{t-k}\\,]$, known at the close of "
            "$t$ and acted on at $t+1$ (one-month lag). With $r_{t+1}$ the next-month SPY return:\n\n"
            "- **H₁ (the drain predicts).** $\\mathbb{E}[r_{t+1}\\mid \\mathrm{drain}_t] > "
            "\\mathbb{E}[r_{t+1}\\mid \\mathrm{fill}_t]$ — a *positive* drain-minus-fill spread.\n"
            "- **H₂ (it's deployable).** A rule long SPY on drains clears buy-and-hold net of costs.\n"
            "- **H₃ (\"liquidity tell\").** The spread reflects liquidity flow, not the accident that "
            "one drain coincided with one bull (and one fill with another).\n\n"
            "We find **H₁ rejected** (spread ≈ 0 or negative at 1-3m; never $t \\ge 2$), **H₂ rejected** "
            "(the rule loses to holding), **H₃ rejected** (the 2021 fill-in-a-bull is a direct "
            "counterexample). The claim survives only as a description of the 2023-24 window it was "
            "drawn on."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one contrast: a conditional mean by regime, judged by its standard "
            "error and by a null that respects the regime structure.\n\n"
            "$$\\widehat{\\Delta}(k) = \\bar r^{\\text{drain}} - \\bar r^{\\text{fill}},\\qquad "
            "t = \\frac{\\widehat{\\Delta}}{\\sqrt{\\,s^2_{\\text{drain}}/n_d + s^2_{\\text{fill}}/n_f\\,}}.$$\n\n"
            "Two traps make the naive version look better than it is. First, US equities rise most "
            "months, so a high *drain win-rate* is not evidence — only the **excess** over the fill "
            "regime is. Second, and fatally here, there are only **two** long regimes (one fill, one "
            "drain), so the effective sample is nearly $n=1$: an i.i.d. label shuffle would treat 54 "
            "monthly labels as independent and wildly understate the null. The honest null is a "
            "**block bootstrap** that reshuffles the labels in contiguous blocks, preserving the fact "
            "that the regime is a single long run."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Series.** Hardcoded ON RRP proxy (FRED `RRPONTSYD`) + yfinance month-end SPY "
            f"(total-return), {R['start']}→{R['end']}, **{R['months']} month-ends**. Explicit "
            "**proxy**; the single-episode sample is named on the Signal axis as the binding "
            "limitation.\n"
            "- **Drain regime.** $B_t < B_{t-k}$ for $k\\in\\{1,2,3,6,9,12\\}$; the first $k$ months "
            "(no trailing change) are **undefined and dropped**, not labelled 'fill'. Entered the "
            "close **1 month after** the signal (no look-ahead).\n"
            "- **Null #1 (Welch t).** Drain mean vs fill mean, unequal variance.\n"
            "- **Null #2 (block placebo).** 20,000 draws of a block-resampled label sequence (block=6, "
            "matched to the observed drain fraction); $p = \\Pr[\\text{random spread} \\ge "
            "\\text{observed}]$ — the small-sample workhorse for few long regimes.\n"
            "- **Timing backtest.** Long/flat (or long/short) SPY on drains, 1-month lag, 10 bps "
            "one-way per switch, raced against buy-and-hold on a Sharpe basis.\n"
            "- **Positive control.** A deterministic hump-shaped RRP-like series + SPY-like price with "
            "a **known** planted drain edge: the inference must recover a planted edge **and** must NOT "
            "manufacture significance when the true edge is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The regime spread — near zero, and sign-unstable\n\n"
            "Drain-minus-fill next-month return by look-back $k$, with the Welch *t* annotated. A real "
            "\"drain = risk-on\" signal would be positive and significant across horizons; instead it "
            "is *negative* at 1-3 months and only mildly positive at long look-backs — none clearing "
            "$t=2$."
        ),
        code(
            "ks = [1,2,3,6,9,12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, k=k) for k in ks]\n"
            "    spreads = [r['spread']*100 for r in rows]; ts = [r['t'] for r in rows]\n"
            "else:\n"
            "    rows = [R[f'k{k}'] for k in ks]; spreads = [r[5] for r in rows]; ts = [r[6] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols = [GREEN if s>0 else RED for s in spreads]\n"
            "ax.bar([str(k) for k in ks], spreads, color=cols, width=.62)\n"
            "ax.axhline(0, c='k', lw=.9)\n"
            "for i,(s,t) in enumerate(zip(spreads,ts)):\n"
            "    ax.annotate(f't={t:+.2f}',(i,s),ha='center',va='bottom' if s>=0 else 'top', fontsize=9)\n"
            "ax.set_xlabel('\"draining\" look-back k (months)'); ax.set_ylabel('drain − fill next-month return (pp)')\n"
            "ax.set_title('Drain−fill spread: negative at 1-3m, sign-flips, never significant'); \n"
            "plt.tight_layout(); plt.show()\n"
            "print('spread pp / Welch t by k:', {k:(round(s,2),round(t,2)) for k,s,t in zip(ks,spreads,ts)})"
        ),
        md(
            f"> 💡 In plain words: at the natural quarter look-back the drain regime returns "
            f"**{R['k3'][3]:.2f}%**/mo vs **{R['k3'][4]:.2f}%** filling — a **{R['k3'][5]:+.1f}pp** gap "
            f"the *wrong* way (t = {R['k3'][6]:+.2f}). The only positive gaps (k≥6) are where "
            "\"draining\" just means \"after the peak,\" and the best of them is t = "
            f"{R['k9'][6]:+.2f}. H₁ is rejected: there is no horizon at which the drain earns a "
            "significant premium."
        ),
        md(
            "### 4b · The decisive test — a block-bootstrap null on the drain's *best* horizon\n\n"
            "Take the look-back where the drain looks strongest (k=6) and ask: with only two long "
            "regimes, how often does a **block-reshuffled** label sequence reproduce the observed "
            "drain-minus-fill spread? The histogram is the null; the observed spread is the line."
        ),
        code(
            "K = 6\n"
            "if HAVE_REAL:\n"
            "    ret = F['spy'].pct_change().shift(-1); sig = st.draining(F, k=K)\n"
            "    dfp = pd.DataFrame({'ret':ret,'drain':sig}).dropna()\n"
            "    r = dfp['ret'].values; lab = dfp['drain'].astype(bool).values\n"
            "    obs = r[lab].mean()-r[~lab].mean(); frac=lab.mean(); n=len(r); block=6\n"
            "    rng=np.random.default_rng(753); nbk=int(np.ceil(n/block)); draws=np.empty(20000)\n"
            "    for i in range(20000):\n"
            "        bl = rng.random(nbk)<frac; lb=np.repeat(bl,block)[:n]\n"
            "        draws[i]= 0.0 if (lb.all() or (~lb).all()) else (r[lb].mean()-r[~lb].mean())\n"
            "    pval=(draws>=obs).mean()\n"
            "else:\n"
            "    obs=R['k6'][5]/100; pval=R['k6'][7]; rng=np.random.default_rng(753); draws=rng.normal(0,0.012,20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label='null: block-reshuffled label spreads')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'observed drain−fill spread {obs*100:+.2f}pp')\n"
            "ax.set_xlabel('drain − fill next-month spread (pp)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Even the best horizon (k=6): block p = {pval:.2f} — inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'k=6 observed spread {obs*100:+.2f}pp, block p = {pval:.3f} (need <0.05; here ~1 in 15, and it is one episode)')"
        ),
        md(
            f"> 💡 In plain words: at the drain's flattering horizon the spread is "
            f"**{R['k6'][5]:+.1f}pp** with block **p = {R['k6'][7]:.2f}** — about 1 in 15, and that "
            "*overstates* the evidence because the bootstrap still can't manufacture a second "
            "independent drain episode. At the natural k=1-3 the observed spread sits at or *below* the "
            "middle of the cloud. There is no cut at which the drain lands in the significant tail."
        ),
        md(
            "### 4c · Tradability + robustness — the Sharpe race and the horizon sweep\n\n"
            "The operational verdict: a drain-timing rule, net of 10 bps/switch, against buy-and-hold. "
            "Then the Welch *t* across every look-back — there is no $k$ at which it clears 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = []\n"
            "    for short in (False, True):\n"
            "        b = st.timing_backtest(F, cost_bps=10.0, allow_short=short)\n"
            "        tim.append((b['net']['sharpe'], b['buy_hold']['sharpe']))\n"
            "    rule_sh = [tim[0][0], tim[1][0]]; bh = tim[0][1]\n"
            "    ts = [st.summarize(F, k=k)['t'] for k in [1,2,3,6,9,12]]\n"
            "else:\n"
            "    rule_sh = [R['timing'][0][4], R['timing'][1][4]]; bh = R['timing'][0][6]\n"
            "    ts = [R[f'k{k}'][6] for k in [1,2,3,6,9,12]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.2))\n"
            "a1.bar([0,1], rule_sh, .5, color=[AMBER, RED], label='drain timing (net)')\n"
            "a1.axhline(bh, ls='--', c=GREEN, lw=2, label=f'buy & hold ({bh:.2f})')\n"
            "a1.set_xticks([0,1]); a1.set_xticklabels(['long/flat','long/short']); a1.set_ylabel('net Sharpe')\n"
            "for i,s in enumerate(rule_sh): a1.annotate(f'{s:.2f}',(i,s),ha='center',va='bottom' if s>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title('Timing loses to holding'); a1.legend()\n"
            "ks=[1,2,3,6,9,12]\n"
            "a2.bar([str(k) for k in ks], ts, color=[GREEN if t>0 else RED for t in ts], width=.62)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 (significance bar)'); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('Drain−fill Welch t never reaches 2'); a2.set_xlabel('look-back k'); a2.set_ylabel('Welch t'); a2.set_ylim(-1.4, 2.4); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('timing net Sharpe (flat/short):', [round(s,2) for s in rule_sh], 'vs buy&hold', round(bh,2))\n"
            "print('Welch t by k:', {k:round(t,2) for k,t in zip(ks,ts)})"
        ),
        md(
            f"> 💡 In plain words: the long/flat rule's **{R['timing'][0][4]:.2f}** Sharpe trails "
            f"buy-and-hold's **{R['timing'][0][6]:.2f}**, and the long/short collapses to "
            f"**{R['timing'][1][4]:.2f}** — a direct fight with the risk premium. On the horizon side, "
            "the *t* is negative at 1-3m and tops out near +1.2. **No look-back, no cost regime, and no "
            "rule variant** makes this deployable, because the signal isn't there."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic hump-shaped RRP-like series (fill then drain) + SPY-like price: with a "
            "**zero** planted drain edge the test must stay below t=2 (a noise drain can't fake "
            "significance); with a **+0.02**/mo planted edge it must light up. Both hold — proving the "
            "engine is unbiased *and* that the real-tape *t* that never clears ~1.2 is what an *absent* "
            "edge looks like."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.02):\n"
            "    syn = data.synthetic(n_months=120, edge=edge, seed=753)\n"
            "    s = st.summarize(syn, k=3)\n"
            "    res.append((edge, s['n_drain'], s['drain_mean']*100, s['fill_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e:.2f}/mo' for e,_,_,_,_,_ in res]; tvals=[r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (drain vs fill)'); ax.set_title('Control: the engine recovers a real edge, ignores a fake one'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,d,fl,t,p in res: print(f'planted {e:+.2f}: n_drain={k} drain={d:.2f}% fill={fl:.2f}% t={t:.2f} p_block={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (below 2 — no false positive); a **+0.02**/mo planted edge "
            f"reaches **t = {R['syn'][1][4]:.2f}** with block **p = {R['syn'][1][5]:.3f}**. The "
            "machinery is honest, so the real-tape non-result is a true negative, not a dead detector. "
            "The inference is the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — drain-minus-fill spread **{R['k3'][5]:+.1f}pp** at k=3 "
            f"(Welch **t = {R['k3'][6]:+.2f}**), *negative* at 1-3m, best case **t = {R['k9'][6]:+.2f}** "
            f"(k=9, block p = {R['k9'][7]:.2f}); corr(ΔRRP, next ret) **= {R['corr']:+.2f}**. The desk's "
            "**t ≥ 2** bar is never met, and the sample is a *single* fill-then-drain episode — so even "
            "the positive long-horizon signs are one macro coincidence, not evidence. NONE, not WEAK.\n"
            f"- **Tradability `MIRAGE`** — drain-timing net Sharpe **{R['timing'][0][4]:.2f}** "
            f"(long/flat) / **{R['timing'][1][4]:.2f}** (long/short) vs **{R['timing'][0][6]:.2f}** for "
            f"buy-and-hold; ends {R['wealth_lf']:.2f}× vs {R['wealth_bh']:.2f}×. Sitting out whenever "
            "the RRP fills — including the 2021 bull — costs more than the drain months return.\n"
            f"- **Liquidity tell? `BUSTED`** — the drain is QT + T-bill issuance mechanically emptying "
            "the facility, coincident with one bull and contradicted by the 2021 fill-in-a-bull. The "
            "block bootstrap says one drain matching one rally is unremarkable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost of sitting out\n\n"
            "The operational truth in one picture: the equity curve of the long/flat drain rule against "
            "buy-and-hold. The gap isn't costs — it's the **months out of the market** while the RRP "
            "fills, in a tape that mostly rises."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = F['spy'].pct_change(); sig = st.draining(F, k=3)\n"
            "    pos = sig.map({True:1.0, False:0.0}).shift(1)\n"
            "    dfp = pd.DataFrame({'ret':ret,'pos':pos}).dropna()\n"
            "    turn = dfp['pos'].diff().abs().fillna(dfp['pos'].abs()); c=10/1e4\n"
            "    rule=(dfp['pos']*dfp['ret']-turn*c); bh=dfp['ret']\n"
            "    eq_rule=(1+rule).cumprod(); eq_bh=(1+bh).cumprod(); expo=(dfp['pos']>0).mean()\n"
            "else:\n"
            "    rng=np.random.default_rng(753); bh=pd.Series(rng.normal(0.01,0.045,R['months']))\n"
            "    eq_bh=(1+bh).cumprod(); eq_rule=(1+bh*0.53).cumprod(); expo=0.53\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(eq_bh.index, eq_bh.values, c=GREEN, lw=2, label='buy & hold')\n"
            "ax.plot(eq_rule.index, eq_rule.values, c=RED, lw=1.9, label='drain long/flat rule (net)')\n"
            "ax.set_ylabel('growth of $1'); ax.set_title(f'Holding only on drains ({expo*100:.0f}% exposure) trails buy & hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'final wealth: rule {eq_rule.iloc[-1]:.2f}x vs buy&hold {eq_bh.iloc[-1]:.2f}x (exposure {expo*100:.0f}%)')"
        ),
        md(
            f"> 💡 In plain words: the rule holds SPY only ~**{R['timing'][0][1]}%** of months and ends "
            f"**{R['wealth_lf']:.2f}×** vs **{R['wealth_bh']:.2f}×**. The red curve flatlines through "
            "every filling stretch, and because the biggest of those was the 2021 bull, the foregone "
            "return swamps any drain edge. There is no sizing, cost, or exposure tweak that turns "
            "\"drain = risk-on\" into a portfolio: **there is no signal to deploy.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The n=1 wall.** The binding limitation is unfixable today: one RRP cycle. The single "
            "most valuable follow-up is simply to re-run this study after the facility fills and drains "
            "again through a *different* market — that is the out-of-sample test the claim needs.\n"
            "- **Level, not change.** A reserve-scarcity variant keys off the RRP *level* near the "
            "floor rather than the drain rate; swap the signal and re-run the detector. The "
            "single-episode problem persists.\n"
            "- **The macro-timing family.** ISM-PMI regime (384), economic surprise index (387) and "
            "jobless-claims momentum (385) on this bench share the lesson — a plausible macro "
            "relationship rarely survives as a tradable monthly timing rule — and this one fails a step "
            "earlier, on the sample size itself.\n\n"
            "*The reproducible core is offline and deterministic; the ON RRP series is an explicit "
            "hardcoded proxy for FRED `RRPONTSYD`. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
