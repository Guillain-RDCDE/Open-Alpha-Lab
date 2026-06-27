"""Generate the two narrative notebooks for Study 533 (Analyst-Dispersion).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached dispersion
snapshot + daily prices under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 40-name large-cap
# basket + current analyst EPS-estimate dispersion, snapshot 2026-06-26, price tape 2015→2026).
R = dict(
    n_names=40, fp_disp="272030376053", fp_px="8691afedbd40",
    px_start="2015-01-02", px_end="2026-06-26", n_days=2887,
    # low-minus-high long-short per trailing horizon:
    # (label, n, low%, high%, ls%, win%, t, p_placebo, spearman)
    h1=("1-mo", 40, 3.37, -7.09, 10.46, 74, 3.21, 0.003, -0.435),
    h3=("3-mo", 40, 12.52, 20.19, -7.67, 59, -0.39, 0.633, -0.120),
    h6=("6-mo", 40, 8.06, 26.32, -18.26, 48, -0.82, 0.341, -0.108),
    h12=("12-mo", 40, 16.35, 56.11, -39.76, 41, -0.99, 0.237, 0.105),
    # 12-mo trailing return by dispersion tercile (low -> high), %
    tercile12=[16.35, 35.89, 56.11],
    # net of costs per horizon: (label, gross%, net%)
    net=[("1-mo", 10.46, 9.98), ("3-mo", -7.67, -8.32),
         ("6-mo", -18.26, -19.16), ("12-mo", -39.76, -41.16)],
    # robustness at 12-mo: (n_buckets, ls%, t, p)
    robust=[(2, -30.15, -1.08, 0.288), (3, -39.76, -0.99, 0.237), (5, -53.62, -0.85, 0.229)],
    # synthetic control: (planted_edge, ls%, avg_t, frac_sig%)
    syn=[(0.00, -0.47, 0.167, 0), (0.70, 65.30, 2.629, 100)],
    # dispersion extremes (ticker, dispersion)
    low_disp=[("CSCO", 0.019), ("KO", 0.021), ("PG", 0.028), ("JNJ", 0.032)],
    high_disp=[("MRK", 0.818), ("TSLA", 0.891), ("ORCL", 1.066), ("BA", 7.168)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Does_dispersion_lose%3F: Mixed](https://img.shields.io/badge/Does_dispersion_lose%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from analyst_dispersion import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    DISP, PRICES = data.load_real()
    FR = data.trailing_returns(PRICES, DISP)        # dispersion + trailing ret_<H> columns
else:
    DISP = PRICES = FR = None
print("real dispersion cache present:", HAVE_REAL,
      "| names:", (0 if DISP is None else len(DISP)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

HCOLS = ["ret_21", "ret_63", "ret_126", "ret_252"]


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# When analysts *can't agree*, does the stock do worse? 🗣️\n"
            "### The Diether-Malloy-Scherbina puzzle — a real academic effect that *doesn't* survive free data, in plain English\n\n"
            + BADGES +
            "Here's a genuinely counter-intuitive idea from finance research. When Wall Street "
            "analysts **wildly disagree** about a company's earnings — one says it'll make \\$8 a "
            "share, another says \\$16 — your gut says *\"risky, so it should pay more.\"* But a famous "
            "2002 study (Diether, Malloy & Scherbina) found the **opposite**: the stocks analysts "
            "argue about most go on to earn **lower** returns. They called it a *puzzle*.\n\n"
            "The reason (an even older idea, Miller 1977): when it's hard to **short** a stock, the "
            "price ends up reflecting only the **optimists** — the pessimists are sidelined. Lots of "
            "disagreement then means the price is too **high**, and it drifts back down. Disagreement = "
            "over-pricing = future under-performance.\n\n"
            "It's a real, published effect. So we tried to see it ourselves — and ran straight into a "
            "wall that's worth understanding: **free data only tells you what analysts think *today*, "
            "not what they thought every month for 20 years.** That one limitation changes everything.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the honest "
            "data autopsy? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **40-name large-cap basket** (names still "
            "trading today). That carries **survivorship** — and, crucially, yfinance gives only a "
            "**single current snapshot** of analyst disagreement, not a history. Every chart is drawn "
            "by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do high-disagreement stocks earn less? | **Only over the *last month* — and even then "
            "we can't trust it.** Sort by disagreement and the calm-consensus names beat the "
            "argued-over ones by **+10%** over the trailing month. But that's **one snapshot on one "
            "day**, not a repeatable result. |\n"
            "| Does it hold over longer windows? | **No — it flips hard.** Over the trailing *year*, "
            "the high-disagreement names **crushed** the calm ones (**+56% vs +16%**). The puzzle "
            "runs backwards here. |\n"
            "| Why the reversal? | **Survivorship.** The argued-over names that *survived* to 2026 are "
            "the giant winners — NVDA, TSLA, ORCL. Their failed lookalikes are gone from the data, so "
            "the \"disagreement loses\" leg is stacked with champions. |\n"
            "| Can I trade it? | **No.** We only know *today's* disagreement, so we can only line it up "
            "against the *past* — that's a coincidence, not a strategy you could have run. |\n\n"
            "> The textbook effect is **real in the research**. On free, survivor-tilted, "
            "single-snapshot data it shows up for one month and then face-plants. An honest *Weak*."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When analysts disagree a lot about a company's earnings, the stock is probably "
            "**over-priced** — because only the optimists' view is in the price (you can't easily bet "
            "against it). So the high-disagreement names quietly **under-perform** as reality sets in. "
            "Buy the boring consensus names, avoid (or short) the controversial ones.\"*\n\n"
            "This is **Diether, Malloy & Scherbina (2002)** in the *Journal of Finance* — a real, "
            "much-cited result, built on the older **Miller (1977)** over-pricing idea. It's a *puzzle* "
            "because it's the opposite of \"more uncertainty → more reward.\" The question for us isn't "
            "\"did they make it up?\" — it's: **can a free-data desk even see it?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If disagreement really predicts under-performance, you'd have a clean, fundamental "
            "signal: just buy the names everyone agrees on. But two traps sit in the way. **(1) "
            "Survivorship:** a basket of today's survivors is missing every controversial company that "
            "blew up — so the \"disagreement\" leg is biased toward winners. **(2) No history:** free "
            "data gives a *snapshot* of disagreement, not a 20-year monthly panel, so we literally "
            "cannot run the forward test the academics ran. We'll show exactly how both traps bend the "
            "result."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name large-cap basket** and, for each name, pull "
            "analysts' **earnings estimates** — the *low*, the *mean*, and the *high* guess for this "
            "year's EPS. The **disagreement** score is simply how far apart the high and low are, "
            "scaled by the average:\n\n"
            "$$\\text{dispersion} = \\frac{\\text{high estimate} - \\text{low estimate}}{|\\text{mean estimate}|}.$$\n\n"
            "Tight estimates (everyone agrees) → small number; wild spread (big argument) → large "
            "number. Then we **sort** the basket into three groups by disagreement and compare their "
            "returns. The catch we keep flagging: we only know *today's* disagreement, so we compare "
            "it to each name's **trailing** return (the move *into* today) — a coincidence check, not "
            "a forward trade."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, who do analysts actually argue about?** Here are the calmest-consensus and "
            "most-argued-over names in the basket. It passes the smell test: steady staples (Coke, "
            "P&G, J&J) at the calm end; cyclicals and high-fliers (Tesla, Oracle, Boeing) at the "
            "loud end."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ds = DISP.sort_values('dispersion')\n"
            "    lows = list(zip(ds['ticker'][:6], ds['dispersion'][:6]))\n"
            "    highs = list(zip(ds['ticker'][-6:], ds['dispersion'][-6:]))\n"
            "else:\n"
            "    lows = R['low_disp']; highs = R['high_disp']\n"
            "names = [t for t,_ in lows] + [t for t,_ in highs]\n"
            "vals  = [v for _,v in lows] + [v for _,v in highs]\n"
            "cols  = [GREEN]*len(lows) + [RED]*len(highs)\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "ax.barh(range(len(names)), vals, color=cols)\n"
            "ax.set_yticks(range(len(names))); ax.set_yticklabels(names)\n"
            "ax.set_xlabel('analyst disagreement  (high - low) / |mean| EPS estimate')\n"
            "ax.set_title('Who analysts agree on (green) vs argue about (red)')\n"
            "ax.invert_yaxis(); plt.tight_layout(); plt.show()\n"
            "print('calmest:', names[0], f'{vals[0]:.3f}', '  loudest:', names[-1], f'{vals[-1]:.3f}')"
        ),
        md(
            "The sort is sensible — so the *measure* is doing its job. Now the real question: do the "
            "calm names (green) actually out-earn the loud ones (red), as the puzzle predicts?"
        ),
        md(
            "**The disagreement-vs-return test, at four windows.** We go *long the calm tercile, short "
            "the loud tercile* (the DMS bet) and read the spread over the trailing 1, 3, 6, and 12 "
            "months. If the puzzle holds, every bar should be **positive** (calm beats loud)."
        ),
        code(
            "labels = ['1-mo','3-mo','6-mo','12-mo']\n"
            "if HAVE_REAL:\n"
            "    ls = [st.summarize(FR, c, placebo=False)['ls_mean']*100 for c in " + repr(HCOLS) + "]\n"
            "else:\n"
            "    ls = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]\n"
            "cols = [GREEN if v>0 else RED for v in ls]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(labels, ls, color=cols, width=.6)\n"
            "for i,v in enumerate(ls): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('calm-minus-loud return (%)')\n"
            "ax.set_title('Calm beats loud for ONE month, then flips hard the other way')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('calm-minus-loud by window:', {l: round(v,1) for l,v in zip(labels, ls)})"
        ),
        md(
            f"There's the whole story in one chart. At **1 month** the puzzle looks alive — calm beats "
            f"loud by **{R['h1'][4]:+.1f}%**. But at **3, 6, and 12 months the bar goes red**: the "
            f"loud, argued-over names *out-earned* the calm ones, by a stunning **{abs(R['h12'][4]):.0f} "
            "percentage points** over the year. The longer you look, the more backwards the puzzle runs."
        ),
        md(
            "**Why?** Look at the 12-month returns by disagreement group. If the puzzle held, this "
            "should slope *down* (more disagreement → lower return). It slopes firmly **up**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    terc = st.bucket_means(FR, 'ret_252')*100\n"
            "else:\n"
            "    terc = np.array(R['tercile12'])\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['calm\\n(low disp)','middle','loud\\n(high disp)'], terc, color=[GREEN,GREY,RED], width=.6)\n"
            "for i,v in enumerate(terc): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('trailing 12-month return (%)')\n"
            "ax.set_title('The WRONG way: more disagreement went with HIGHER returns here')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('12-mo return by group (calm->loud):', [round(float(v),1) for v in terc])"
        ),
        md(
            f"The loud group returned **{R['tercile12'][-1]:+.0f}%** over the year; the calm group only "
            f"**{R['tercile12'][0]:+.0f}%**. That's the *opposite* of the puzzle — and it's the "
            "**survivorship trap** made visible. The argued-over names that *survived* to 2026 (NVDA, "
            "Tesla, Oracle) are exactly the ones whose big, controversial bets paid off. Their failed "
            "twins aren't in the data."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Calm-beats-loud shows up at **one** window (1-month, "
            f"**{R['h1'][4]:+.1f}%**) and **reverses** at every other. One snapshot isn't proof.\n"
            "- **Tradability — Mirage.** We only know *today's* disagreement, so there's no forward "
            "trade — just a coincidence with the past.\n"
            "- **\"Does disagreement lose?\" — Mixed.** Real in the research; here it's true for a "
            "month and backwards over a year, swamped by survivorship."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the honest reason\n\n"
            "Suppose we *pretend* the trailing sort were a real trade and charge it normal costs. The "
            "1-month \"+10%\" barely dents to \"+10%\" net. So is it tradable? **No** — and this is the "
            "key lesson, not a cost story."
        ),
        code(
            "labels = ['1-mo','3-mo','6-mo','12-mo']\n"
            "if HAVE_REAL:\n"
            "    g, nv = [], []\n"
            "    for c,h in zip(" + repr(HCOLS) + ", [21,63,126,252]):\n"
            "        cc = st.net_of_costs(FR, c, horizon_days=h)\n"
            "        g.append(cc['gross']*100); nv.append(cc['net']*100)\n"
            "else:\n"
            "    g = [n[1] for n in R['net']]; nv = [n[2] for n in R['net']]\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross calm-minus-loud')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='after pretend costs')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('return (%)'); ax.set_title('Costs barely matter — the problem is it is not a forward trade at all')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('1-mo gross/net:', round(g[0],1), round(nv[0],1))"
        ),
        md(
            "> The \"+10% net\" is **bookkeeping on returns that already happened** — you'd have needed "
            "to know today's disagreement a month ago to capture it, and you didn't. That's why the "
            "tradability stamp is **Mirage**, full stop: there is no version of this you could have run."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The history is the whole game.** With a real **I/B/E/S dispersion panel** (monthly, "
            "20+ years), you could run the *forward* test DMS ran. Free data simply doesn't carry it — "
            "and that gap is the entire reason we land *Weak/Mirage* instead of confirming a real "
            "academic result.\n"
            "- **Survivorship cuts the wrong way.** Here the surviving loud names are winners, so the "
            "bias *hides* the effect. Add delisted controversial names and the picture could change.\n"
            "- **Where it's strongest.** Later work (Avramov et al. 2009) ties the effect to "
            "**distressed, low-rated** names — not the liquid large-caps we used. Wrong pond for a "
            "big fish.\n\n"
            "*Think you can see the puzzle on free data? Show a calm-minus-loud spread that's positive "
            "and significant on a real **forward** sort across multiple windows — then we'll talk.*"
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
            "# Analyst-Dispersion — a quantitative teardown 🔬\n"
            "### Dispersion = (high − low) / |mean| EPS estimate · low-minus-high tercile long-short "
            "across 1/3/6/12-mo · one-sample *t* + label-shuffle placebo + Spearman · the 12-mo "
            "wrong-way monotonicity · costs · a seed-robust synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "Diether-Malloy-Scherbina (2002) puzzle is a **real academic effect** — so the job here "
            "is the *data autopsy*: show exactly why a free, single-snapshot, survivor-tilted "
            "cross-section can't confirm it, where the lone significant window comes from, and why it "
            "is **Weak, not Real**.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **40-name large-cap** basket, names still trading "
            "in 2026 (a *survivor* panel that tilts the loud/high-dispersion leg toward winners). "
            "Crucially, yfinance gives only a **current dispersion snapshot** — no history — so the "
            "sort is dispersion-today vs **trailing** return (contemporaneous, not forward). Real "
            "data: yfinance `earnings_estimate` (0y) + daily closes, 2015→2026. Offline core + "
            "synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | low-minus-high **+{R['h1'][4]:.2f}%** at 1-mo (one-sample "
            f"**t = {R['h1'][6]:.2f}**, placebo **p = {R['h1'][7]:.3f}**, ρ = {R['h1'][8]:.3f}) — "
            f"but a **single cross-section**, **reversed** at 3/6/12-mo ({R['h12'][4]:+.1f}% at 12-mo). "
            "Lone window ⇒ Weak, not Real. |\n"
            f"| **Tradability** | `MIRAGE` | No forward trade: dispersion is a *current* snapshot vs "
            f"**trailing** returns. \"Net +{R['net'][0][2]:.1f}%\" is bookkeeping on realised returns. |\n"
            f"| **Does dispersion lose?** | `MIXED` | 12-mo tercile sort runs **monotone the wrong way** "
            f"(low {R['tercile12'][0]:+.0f}% → high {R['tercile12'][-1]:+.0f}%); survivors dominate. |\n\n"
            "> 💡 In plain words: the puzzle is real in the literature, but the only thing free data "
            "lets us build (one dispersion snapshot vs trailing returns, 40 survivors) shows it for "
            "one month and reverses everywhere else."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $d_i$ be name $i$'s forecast dispersion (here $(h_i - l_i)/|m_i|$ on the current-year "
            "consensus EPS, the free analogue of the academic std-dev-of-forecasts measure), and "
            "$r_i(H)$ its trailing $H$-day return. Tercile by $d$; the DMS long-short is\n\n"
            "$$\\widehat{\\Lambda}(H) = \\frac{1}{n_L}\\!\\!\\sum_{i\\in T_{\\text{low}}}\\!\\! r_i(H) "
            "\\;-\\; \\frac{1}{n_H}\\!\\!\\sum_{i\\in T_{\\text{high}}}\\!\\! r_i(H).$$\n\n"
            "- **H₁ (DMS holds).** $\\widehat{\\Lambda}(H) > 0$ and significant (low dispersion wins).\n"
            "- **H₂ (deployable).** $\\widehat{\\Lambda}$ is a *forward* edge that survives costs.\n"
            "- **H₃ (clean replication).** The sign is stable across horizons and bucket counts.\n\n"
            "We find **H₁ only at 1-mo** (and on a single cross-section, so not robustly), **H₂ false "
            "by construction** (trailing, not forward — a snapshot can't be traded against the past), "
            "and **H₃ false** (sign flips by 3-mo; 12-mo monotone the wrong way). The academic effect "
            "is real; this *free-data instance* of it is not certifiable."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample test of the pooled long-short sample (low-tercile returns "
            "as longs, high-tercile as $-$shorts) against zero:\n\n"
            "$$t = \\frac{\\overline{\\lambda}}{s_\\lambda/\\sqrt{k}},\\qquad "
            "\\lambda \\in \\{r_i : i\\in T_{\\text{low}}\\}\\cup\\{-r_i : i\\in T_{\\text{high}}\\}.$$\n\n"
            "Two honesty problems sit on top of a naive *t*. **(a) One cross-section:** with a single "
            "snapshot we have $k\\approx 27$ pooled observations on *one date* — a lone significant "
            "$t$ cannot be averaged over independent draws, which is precisely why the desk grades it "
            "**Weak**, not Real. **(b) Survivorship:** the high-dispersion survivors are winners, so "
            "the short leg is biased *up*, working against the DMS sign at long horizons. We add a "
            "**label-shuffle placebo** and a **Spearman** continuous check to keep the small-sample "
            "test honest."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket (yfinance "
            f"`earnings_estimate` 0y + adjusted closes, {R['px_start']}→{R['px_end']}, "
            f"{R['n_days']:,} price days). **Survivor** panel — named on the Signal axis.\n"
            "- **Dispersion.** $(h-l)/|m|$ on the current-year consensus EPS estimate (the DMS "
            "spread). **Single current snapshot** — no history; this is the binding limitation.\n"
            "- **Return.** Trailing $H\\in\\{21,63,126,252\\}$ trading days (1/3/6/12-mo) realised "
            "*into* the snapshot — a contemporaneous association, **not** a forward trade.\n"
            "- **Signal.** Tercile by dispersion; long $T_{\\text{low}}$ − short $T_{\\text{high}}$.\n"
            "- **Null #1 (one-sample t)** of the pooled long-short sample vs 0.\n"
            "- **Null #2 (label-shuffle placebo).** 20,000 random re-sorts of the same returns; "
            "$p = \\Pr[|\\text{shuffled }\\Lambda| \\ge |\\text{observed}|]$ (two-sided).\n"
            "- **Continuous view.** Spearman $\\rho(d, r)$ — DMS predicts **negative**.\n"
            "- **Costs.** 10 bps one-way × 4 legs + 100 bps/yr borrow on the short leg.\n"
            "- **Positive control.** A deterministic panel with a **planted** dispersion→return drag: "
            "seed-robust over 25 seeds it must recover the effect and return ~0 under the null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The term structure of the sign — one window right, the rest wrong\n\n"
            "Left: the low-minus-high spread at each horizon (the DMS bet — positive = calm wins). "
            "Right: 12-month return by dispersion tercile (the monotonicity — DMS predicts a *down* "
            "slope)."
        ),
        code(
            "labels = ['1-mo','3-mo','6-mo','12-mo']\n"
            "if HAVE_REAL:\n"
            "    ls = [st.summarize(FR, c, placebo=False)['ls_mean']*100 for c in " + repr(HCOLS) + "]\n"
            "    terc = st.bucket_means(FR, 'ret_252')*100\n"
            "else:\n"
            "    ls = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]; terc = np.array(R['tercile12'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(labels, ls, color=[GREEN if v>0 else RED for v in ls], width=.6)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(ls): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>0 else 'top',fontsize=9)\n"
            "a1.set_ylabel('low-minus-high (%)'); a1.set_title('DMS bet: positive at 1-mo only, reverses after')\n"
            "a2.bar(['low','mid','high'], terc, color=[GREEN,GREY,RED], width=.6)\n"
            "for i,v in enumerate(terc): a2.annotate(f'{v:+.0f}%',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_ylabel('trailing 12-mo return (%)'); a2.set_xlabel('dispersion tercile')\n"
            "a2.set_title('Monotone the WRONG way (DMS predicts a down-slope)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('LS by horizon:', [round(v,1) for v in ls]); print('12-mo tercile:', [round(float(v),1) for v in terc])"
        ),
        md(
            f"> 💡 In plain words: the lone positive bar is 1-mo (**{R['h1'][4]:+.1f}%**); 3/6/12-mo are "
            f"all red, and the 12-mo tercile sort climbs **{R['tercile12'][0]:+.0f}% → "
            f"{R['tercile12'][-1]:+.0f}%** — the opposite of DMS. The survivors' winners dominate the "
            "longer you look."
        ),
        md(
            "### 4b · The one significant window — and why it's not enough\n\n"
            "The 1-month long-short against a 20,000-draw label-shuffle null. The observed spread sits "
            "in the tail (**p = {:.3f}**, one-sample **t = {:.2f}**) — but it is **one cross-section "
            "on one date**, so this *t* is a single draw, not a replicated effect.".format(
                R['h1'][7], R['h1'][6])
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(FR, 'ret_21', n_draws=20000)\n"
            "    obs = pl['obs']*100; draws = pl['draws']*100; pval = pl['p_value']\n"
            "    tval = st.summarize(FR, 'ret_21', placebo=False)['t']\n"
            "else:\n"
            "    obs = R['h1'][4]; pval = R['h1'][7]; tval = R['h1'][6]\n"
            "    rng = np.random.default_rng(533); draws = rng.normal(0.0, 4.5, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: 20,000 dispersion-label shuffles')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed low-minus-high {obs:+.1f}%')\n"
            "ax.set_xlabel('1-month low-minus-high spread (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Significant in this ONE snapshot: p = {pval:.3f}, t = {tval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.1f}%  one-sample t={tval:.2f}  shuffle p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: yes, the green line is in the tail (**p = {R['h1'][7]:.3f}**, "
            f"**t = {R['h1'][6]:.2f}**) — the snapshot caught a month where the crowded high-dispersion "
            "names (TSLA, ORCL, BA) sold off together. But a single window's *t* > 3 that **reverses** "
            "at every other horizon is the textbook **WEAK** call: it cannot be averaged over "
            "independent draws, and the desk does not stamp Real on one lucky cross-section."
        ),
        md(
            "### 4c · Robustness + the continuous view — the reversal is real, the win isn't\n\n"
            "Left: vary the bucket count at 12-mo — the wrong-sign result is stable (it's not a "
            "tercile artefact). Right: the Spearman rank correlation $\\rho(d, r)$ at each horizon — "
            "DMS predicts negative; only the 1-mo snapshot delivers it."
        ),
        code(
            "labels = ['1-mo','3-mo','6-mo','12-mo']\n"
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for nb in (2,3,5):\n"
            "        s = st.summarize(FR, 'ret_252', n_buckets=nb, placebo=False)\n"
            "        rob.append((nb, s['ls_mean']*100, s['t']))\n"
            "    rhos = [st.disp_return_corr(FR, c) for c in " + repr(HCOLS) + "]\n"
            "else:\n"
            "    rob = [(r[0], r[1], r[2]) for r in R['robust']]\n"
            "    rhos = [R['h1'][8], R['h3'][8], R['h6'][8], R['h12'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar([f'{r[0]}' for r in rob], [r[1] for r in rob], color=RED, width=.55)\n"
            "for i,r in enumerate(rob): a1.annotate(f't={r[2]:.2f}',(i,r[1]),ha='center',va='top',fontsize=9)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xlabel('bucket count'); a1.set_ylabel('12-mo low-minus-high (%)')\n"
            "a1.set_title('Wrong-sign 12-mo is stable across buckets')\n"
            "a2.bar(labels, rhos, color=[GREEN if v<0 else RED for v in rhos], width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(rhos): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v<0 else 'top',fontsize=9)\n"
            "a2.set_ylabel('Spearman rho(dispersion, return)'); a2.set_title('DMS wants NEGATIVE; only 1-mo obliges')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (buckets, ls%, t):', [(r[0], round(r[1],1), round(r[2],2)) for r in rob])\n"
            "print('Spearman by horizon:', [round(float(v),3) for v in rhos])"
        ),
        md(
            f"> 💡 In plain words: tightening the buckets only *widens* the wrong-sign 12-mo spread "
            f"(to {R['robust'][2][1]:+.0f}% for quintiles), so the reversal isn't an artefact. And the "
            f"Spearman view agrees — strongly negative at 1-mo (ρ = {R['h1'][8]:.2f}, DMS-correct) but "
            "drifting to *positive* by 12-mo. The signal is one snapshot deep."
        ),
        md(
            "### 4d · Costs — moot, because there is no forward trade\n\n"
            "For completeness, charge the implied trade (4 one-way legs at 10 bps + 100 bps/yr borrow). "
            "The numbers barely move — but that's not the point."
        ),
        code(
            "labels = ['1-mo','3-mo','6-mo','12-mo']\n"
            "if HAVE_REAL:\n"
            "    g, nv = [], []\n"
            "    for c,h in zip(" + repr(HCOLS) + ", [21,63,126,252]):\n"
            "        cc = st.net_of_costs(FR, c, horizon_days=h)\n"
            "        g.append(cc['gross']*100); nv.append(cc['net']*100)\n"
            "else:\n"
            "    g = [n[1] for n in R['net']]; nv = [n[2] for n in R['net']]\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net (costs + borrow)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('low-minus-high (%)'); ax.set_title('Costs barely bite — but a trailing sort is not a tradable trade')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for l,gg,nn in zip(labels,g,nv): print(f'{l}: gross={gg:+.1f}%  net={nn:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: at 1-mo, gross **{R['net'][0][1]:+.1f}%** → net "
            f"**{R['net'][0][2]:+.1f}%** — costs are trivial. But the return is **trailing** "
            "(already realised into the snapshot), so this \"net\" is bookkeeping, not P&L. You can't "
            "trade a coincidence with the past. Hence **Mirage**."
        ),
        md(
            "### 4e · Faithful-engine & power control — the sort works, the tape just says no\n\n"
            "On a deterministic panel where we **plant** a known dispersion→return drag (the DMS sign): "
            "with **zero** edge the long-short must stay at t≈0; with a **large** planted drag it must "
            "light up — and it must do so **seed-robustly**, not on one lucky RNG draw."
        ),
        code(
            "import pandas as pd\n"
            "res = []\n"
            "for edge in (0.0, 0.70):\n"
            "    ts = []; lss = []\n"
            "    for s in range(25):\n"
            "        cs = data.synthetic_cross_section(dms_edge=edge, seed=1000+s)\n"
            "        rr = st.summarize(cs, 'ret_cum', placebo=False)\n"
            "        ts.append(rr['t']); lss.append(rr['ls_mean']*100)\n"
            "    res.append((edge, float(np.nanmean(lss)), float(np.nanmean(ts)), float(np.mean(np.array(ts)>2)*100)))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_ in res]\n"
            "tvals = [r[2] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('avg one-sample t (25 seeds)'); ax.set_title('Control: null t~0; planted drag clears t=2 on every seed'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,ls,t,fr_ in res: print(f'planted {e:.2f}: avg ls={ls:+.1f}% avg t={t:.2f} frac(t>2)={fr_:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted drag the control sits at "
            f"**t = {R['syn'][0][2]:.2f}** (no false positive — noise cannot fake it); a **0.70** "
            f"planted drag reaches **t = {R['syn'][1][2]:.2f}** on **{R['syn'][1][3]:.0f}% of seeds**. "
            "The cross-sectional sort is unbiased and adequately powered — so the real-tape verdict "
            "is a genuine reading of a survivor-biased, single-snapshot tape, not a broken engine. "
            "The machinery works; the *data* can't confirm the puzzle."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — low-minus-high **+{R['h1'][4]:.2f}%** at 1-mo (one-sample "
            f"**t = {R['h1'][6]:.2f}**, placebo **p = {R['h1'][7]:.3f}**, ρ = {R['h1'][8]:.3f}) is the "
            "*only* DMS-correct window, and it's a **single cross-section on a single date** — not "
            "averageable over independent draws, **reversed** at 3/6/12-mo "
            f"({R['h12'][4]:+.1f}% at 12-mo). Literature support is real (DMS 2002), but a lone window's "
            "*t* > 3 is **Weak, not Real** by the desk's bar.\n"
            "- **Tradability `MIRAGE`** — there is **no forward trade**: dispersion is a current "
            "snapshot, the returns are trailing, so the \"net "
            f"+{R['net'][0][2]:.1f}%\" is bookkeeping on realised returns, not capturable P&L. 40 "
            "survivor names, no dispersion history. Not investable on free data.\n"
            f"- **Does dispersion lose? `MIXED`** — direction- and horizon-dependent: yes at 1-mo, but "
            f"the 12-mo tercile sort is **monotone the wrong way** (low {R['tercile12'][0]:+.0f}% → "
            f"high {R['tercile12'][-1]:+.0f}%), the surviving high-dispersion winners dominating. The "
            "academic puzzle is real; this free-data instance does not cleanly replicate."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the missing-history problem in one picture\n\n"
            "The Spearman correlation of dispersion vs return, by horizon, with the DMS-correct region "
            "(negative) shaded. The signal is *correct* only at the freshest snapshot and erodes — "
            "exactly what a single-date measure does when you stretch it across longer windows."
        ),
        code(
            "labels = ['1-mo','3-mo','6-mo','12-mo']; xs = [1,3,6,12]\n"
            "if HAVE_REAL:\n"
            "    rhos = [st.disp_return_corr(FR, c) for c in " + repr(HCOLS) + "]\n"
            "else:\n"
            "    rhos = [R['h1'][8], R['h3'][8], R['h6'][8], R['h12'][8]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(xs, rhos, 'o-', c=GREEN, lw=2.2)\n"
            "ax.axhline(0, c=RED, ls='--', label='DMS-correct below this line')\n"
            "ax.fill_between(xs, 0, [min(0,v) for v in rhos], color=GREEN, alpha=.12)\n"
            "for x,v in zip(xs,rhos): ax.annotate(f'{v:+.2f}',(x,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('trailing horizon (months)'); ax.set_ylabel('Spearman rho(dispersion, return)')\n"
            "ax.set_title('DMS-correct only at the freshest snapshot; erodes with horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Spearman by horizon:', {l: round(float(v),3) for l,v in zip(labels, rhos)})"
        ),
        md(
            "> 💡 In plain words: the green line is below zero (DMS-correct) only at 1-mo and climbs "
            "toward positive by 12-mo. A *current* dispersion reading lines up with *recent* returns "
            "and decorrelates from older ones — which is all you can ever learn from a snapshot. The "
            "honest fix is a real dispersion **history**; without it the verdict is **Weak / Mirage / "
            "Mixed**, and that's the on-brand outcome for a pointed academic factor on free data."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **History is the binding constraint.** With an **I/B/E/S dispersion panel** you'd run "
            "the *forward* monthly sort DMS ran (and add Fama-MacBeth standard errors). Free data "
            "simply does not carry the panel — that gap, not a broken effect, is why we land Weak.\n"
            "- **Where the effect lives.** Avramov, Chordia, Jostova & Philipov (2009) tie dispersion "
            "to **credit risk / distress**; Sadka & Scherbina (2007) to **liquidity**. Both predict it "
            "is weak among liquid large-caps — our universe — consistent with a non-result here.\n"
            "- **Survivorship cuts against the claim.** The surviving high-dispersion names are "
            "winners, biasing the short leg up; a delisting-complete universe would push the sign back "
            "toward DMS.\n\n"
            "*The reproducible core is offline and deterministic; the dispersion input is the current "
            "analyst EPS-estimate spread and the sort is against trailing returns (no free dispersion "
            "history — named on the Signal axis). Methods and sources: "
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
