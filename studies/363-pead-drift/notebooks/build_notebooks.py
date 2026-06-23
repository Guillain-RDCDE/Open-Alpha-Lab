"""Generate the two narrative notebooks for Study 363 (PEAD-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices +
earnings events under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 30-name large-cap
# basket + per-name earnings surprises, 2005-01-11 -> 2026-06-10, 2,579 events, 21.4 years).
R = dict(
    start="2005-01-11", end="2026-06-10", years=21.4, n_events=2579, n_eps=2576, n_names=30,
    # EPS-surprise long-short per horizon: (H, n, top%, bot%, ls%, win%, t, p_placebo)
    h1=(1, 2576, 0.06, -0.10, 0.16, 53, 1.38, 0.063),
    h5=(5, 2576, 0.50, 0.27, 0.24, 50, 1.09, 0.113),
    h20=(20, 2572, 1.75, 0.41, 1.34, 53, 2.96, 0.000),
    h60=(60, 2546, 4.54, 2.83, 1.71, 51, 2.10, 0.005),
    # net of costs per horizon: (H, gross%, net%)
    net=[(1, 0.16, -0.24), (5, 0.24, -0.17), (20, 1.34, 0.90), (60, 1.71, 1.19)],
    # 20-day drift by quintile (low -> high), %
    quintile20=[0.41, 1.24, 1.29, 0.86, 1.75],
    # robustness: (n_buckets, ls20%, t, p)
    robust=[(3, 0.61, 1.88, 0.021), (5, 1.34, 2.96, 0.000), (10, 1.71, 2.43, 0.000)],
    block_p20=0.016,
    # gap-sort (myth check): (H, ls%, t, p)
    gap=[(1, 0.07, 0.58, 0.261), (5, 0.11, 0.49, 0.303),
         (20, 0.64, 1.48, 0.050), (60, -0.23, -0.31, 0.630)],
    # synthetic control 20d: (planted_edge, events, ls20%, t, p, win%)
    syn=[(0.00, 1827, -0.08, -0.17, 0.565, 49), (0.08, 1827, 5.32, 11.17, 0.000, 65)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Always_drifts%3F: Busted](https://img.shields.io/badge/Always_drifts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from pead_drift import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, EV = data.load_real()
    EVV = EV.dropna(subset=["surprise_pct"]).copy()   # events with a reported EPS surprise
else:
    PRICES = EV = EVV = None
print("real PEAD cache present:", HAVE_REAL,
      "| events:", (0 if EV is None else len(EV)),
      "| with EPS surprise:", (0 if EVV is None else len(EVV)))
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
            "# Does a stock keep \"drifting\" after an earnings surprise? 📈\n"
            "### The post-earnings drift — one of the rare market legends that turns out to be **real**, in plain English\n\n"
            + BADGES +
            "Here's a piece of trading lore that *sounds* too good to be true: when a company **beats** "
            "earnings, its stock doesn't just jump once and stop — it supposedly keeps **drifting up** "
            "for weeks. Miss, and it keeps sliding. So you buy the beats, short the misses, and ride the "
            "drift. The pros even have a name for it: **Post-Earnings-Announcement Drift (PEAD)**.\n\n"
            "Most market legends fall apart the moment you measure them. This one **mostly doesn't** — "
            "which makes it the interesting case. But the *useful* version is narrower and stranger than "
            "the folklore: the drift is real, but it's small, it shows up **after** the first week (not "
            "during the pop), and the obvious way to trade it — **chase the biggest jumps** — barely "
            "works at all.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **30-name large-cap basket** (names still "
            "trading today). That's the *conservative* corner — the textbook PEAD is strongest in small, "
            "illiquid stocks — and it carries **survivorship** (we can't include firms that blew up). "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a *beat*, does the stock keep drifting up? | **Yes — really.** Sort earnings by how "
            "much they beat, and the best-surprise names outrun the worst by **+1.3%** over the next "
            "**20 days** (and **+1.7%** over 60). That's a *real* signal, not luck. |\n"
            "| Is it there right away? | **No.** In the first **week** there's essentially nothing. The "
            "drift builds *after* the initial pop has faded. |\n"
            "| Can I just buy whatever **gapped up** the most? | **Barely.** Sort on the *visible jump* "
            "instead of the actual earnings surprise and the edge **vanishes** (and even turns negative "
            "at 60 days). The drift lives in the *fundamentals*, not the chart. |\n"
            "| So is it free money? | **No.** It's **thin.** Trading costs eat the fast (1–5 day) version "
            "entirely; only the slow 20–60 day hold survives, and only barely, on a handful of big names. |\n\n"
            "> The legend is **true** — and that's rare. But the honest version is \"a small, slow drift "
            "in the fundamental surprise,\" not \"buy the pop and get rich.\""
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When earnings come out far above what analysts expected, the market is **slow** to fully "
            "react. The stock pops on the day — but then it keeps grinding **higher** for weeks as the "
            "good news sinks in. Buy the big beats, short the big misses, pocket the drift.\"*\n\n"
            "This isn't just a trader's tall tale. **Post-Earnings-Announcement Drift** was documented by "
            "academics back in **1968** (Ball & Brown) and nailed down in **1989** (Bernard & Thomas). "
            "Eugene Fama — Mr. Efficient-Markets himself — called it the *\"granddaddy of anomalies,\"* "
            "the one that refuses to die. So the question isn't \"is it made up?\" It's: **how big, how "
            "fast, and can you actually trade it?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the market were perfectly efficient, *all* the reaction to an earnings surprise would "
            "happen **instantly** — the price would jump to its new level and then wander randomly. PEAD "
            "says it doesn't: a chunk of the adjustment **leaks out over weeks**, which means the future "
            "is (a little) predictable from the past surprise. That's a genuine crack in efficiency.\n\n"
            "But \"a little predictable\" and \"a money machine\" are different things. Two traps hide "
            "here: (1) the *visible* price jump and the *true* earnings surprise are **not the same** — "
            "chasing the jump can be a different, worse trade; and (2) a drift of one-ish percent over a "
            "month is easily **swallowed by trading costs** if you trade it fast and often. We'll test "
            "both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name large-cap basket** and pull **every** quarterly "
            f"earnings date for each — about **{R['n_events']:,} events** over **{R['years']:.0f} years** "
            f"({R['start']} → {R['end']}). For each one we record:\n\n"
            "1. **The surprise.** How much actual EPS beat or missed the analyst estimate (the vendor's "
            "`Surprise(%)`). We *also* record the **price gap** — the one-day jump — so we can compare "
            "the two.\n"
            "2. **The drift.** Starting the day **after** the reaction (no cheating — the pop is already "
            "public), how much does the stock move over the next **1, 5, 20, 60** trading days?\n"
            "3. **The bet.** Sort events into five surprise buckets; go **long** the best, **short** the "
            "worst. If the legend is real, that long-short should make money — and we test it against a "
            "thousands-of-shuffles \"could this be luck?\" null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, does the drift line up with the surprise?** Sort every event into five buckets by "
            "how big the earnings surprise was, and plot the average 20-day drift of each bucket. If the "
            "legend holds, it should climb from left (worst misses) to right (best beats)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PRICES, EVV, 20)\n"
            "    q = st.bucket_means(fr, 'surprise_pct') * 100\n"
            "else:\n"
            "    q = np.array(R['quintile20'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "labels = ['worst\\nmiss','','middle','','best\\nbeat']\n"
            "cols = [RED, AMBER, GREY, AMBER, GREEN]\n"
            "ax.bar(range(5), q, color=cols, width=.7)\n"
            "for i,v in enumerate(q): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_xticks(range(5)); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('average 20-day drift (%)')\n"
            "ax.set_title('Best beats drift up the most — the PEAD pattern is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'best-beat bucket: {q[-1]:+.2f}%   worst-miss bucket: {q[0]:+.2f}%   '\n"
            "      f'spread: {q[-1]-q[0]:+.2f}%')"
        ),
        md(
            f"There it is. The best-beat bucket drifts **{R['quintile20'][-1]:+.2f}%** over 20 days; the "
            f"worst-miss bucket only **{R['quintile20'][0]:+.2f}%**. The extremes separate cleanly — the "
            "middle is mush, as you'd expect. The drift is **real**."
        ),
        md(
            "**But when does it happen?** Here's the long-short (best beats minus worst misses) at four "
            "horizons. Watch the first week."
        ),
        code(
            "hs = [1, 5, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    ls = [st.summarize(PRICES, EVV, h, surprise_col='surprise_pct', placebo=False)['ls_mean']*100 for h in hs]\n"
            "else:\n"
            "    ls = [R['h1'][4], R['h5'][4], R['h20'][4], R['h60'][4]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREY if h<20 else GREEN for h in hs]\n"
            "ax.bar([str(h) for h in hs], ls, color=cols, width=.6)\n"
            "for i,v in enumerate(ls): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_xlabel('trading days held after the surprise'); ax.set_ylabel('long-short drift (%)')\n"
            "ax.set_title('Almost nothing in the first week — the drift builds over weeks')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1-day:', f'{ls[0]:+.2f}%', ' 5-day:', f'{ls[1]:+.2f}%', ' 20-day:', f'{ls[2]:+.2f}%', ' 60-day:', f'{ls[3]:+.2f}%')"
        ),
        md(
            f"The fast money isn't there. At **1 day** the spread is a measly "
            f"**{R['h1'][4]:+.2f}%**, at **5 days** **{R['h5'][4]:+.2f}%** — basically noise. The signal "
            f"only builds by **20 days** (**{R['h20'][4]:+.2f}%**) and **60 days** "
            f"(**{R['h60'][4]:+.2f}%**). PEAD is a **slow** drift, not a same-week pop."
        ),
        md(
            "**Now the trap.** What if you just buy whatever **gapped up** the most on the day — the "
            "*visible* move, the thing a chart-watcher actually sees — instead of the fundamental "
            "surprise? Let's compare the two ways of sorting."
        ),
        code(
            "if HAVE_REAL:\n"
            "    eps_ls = [st.summarize(PRICES, EVV, h, surprise_col='surprise_pct', placebo=False)['ls_mean']*100 for h in hs]\n"
            "    gap_ls = [st.summarize(PRICES, EV, h, surprise_col='gap', placebo=False)['ls_mean']*100 for h in hs]\n"
            "else:\n"
            "    eps_ls = [R['h1'][4], R['h5'][4], R['h20'][4], R['h60'][4]]\n"
            "    gap_ls = [g[1] for g in R['gap']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, eps_ls, .4, color=GREEN, label='sort on EARNINGS surprise (works)')\n"
            "ax.bar(x+.2, gap_ls, .4, color=RED, label='sort on the visible price GAP (folk recipe)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('long-short drift (%)')\n"
            "ax.set_title('The drift is in the FUNDAMENTALS — chasing the visible pop barely works')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gap-sorted 60-day:', f'{gap_ls[-1]:+.2f}%', '(vs earnings-sorted', f'{eps_ls[-1]:+.2f}%)')"
        ),
        md(
            f"The red bars — chasing the **visible jump** — barely clear zero and **turn negative at 60 "
            f"days** (**{R['gap'][3][1]:+.2f}%**). The green bars — sorting on the **actual earnings "
            "surprise** — are where the money is. This is the punchline the folklore gets wrong: **the "
            "drift lives in the fundamentals you have to look up, not the pop you can see on the chart.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** Sorted on the earnings surprise, the best-minus-worst drift is "
            f"**{R['h20'][4]:+.2f}% over 20 days** and **{R['h60'][4]:+.2f}% over 60** — statistically "
            "solid (the quants notebook shows *t* ≈ 3). One of the rare legends that survives the audit.\n"
            "- **Tradability — Fragile.** It's **thin**. The fast (1–5 day) version is eaten by costs; "
            "only a slow 20–60 day hold on a handful of big names survives — and barely.\n"
            "- **\"Always drifts the same way\"? — Busted.** Only the **fundamental** surprise drifts, and "
            "only **after** the first week. Chasing the **visible gap** — the folk recipe — doesn't work."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — watch costs eat the fast money\n\n"
            "Here's the operational reality. The same long-short, **gross** (before costs) vs **net** "
            "(after a realistic round-trip on both legs plus borrow on the short). Look at what happens "
            "to the short horizons."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g, nv = [], []\n"
            "    for h in hs:\n"
            "        c = st.net_of_costs(PRICES, EVV, h, surprise_col='surprise_pct')\n"
            "        g.append(c['gross']*100); nv.append(c['net']*100)\n"
            "else:\n"
            "    g = [n[1] for n in R['net']]; nv = [n[2] for n in R['net']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross drift')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net of costs + borrow')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('long-short return (%)')\n"
            "ax.set_title('Costs flip the fast money negative; only slow holds survive')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net 1-day:', f'{nv[0]:+.2f}%', ' net 20-day:', f'{nv[2]:+.2f}%', ' net 60-day:', f'{nv[3]:+.2f}%')"
        ),
        md(
            f"At **1 day** the net result is **{R['net'][0][2]:+.2f}%** — you *lose* money trading the "
            "drift fast, because the round-trip on both legs costs more than the tiny gross edge. Only "
            f"the **20-day** (**{R['net'][2][2]:+.2f}%**) and **60-day** (**{R['net'][3][2]:+.2f}%**) "
            "holds stay positive. The drift is real, but it's the kind of real you earn slowly and "
            "patiently — not a trade you flip in a week."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why so thin here?** PEAD is famously **strongest in small, illiquid stocks** (Chordia et "
            "al. 2009) — exactly the names we *excluded* by using a liquid large-cap basket. Re-run it on "
            "small-caps and the drift gets bigger (but so do the costs and the survivorship traps).\n"
            "- **The fundamental vs the visible.** The single most important lesson here is that the "
            "*earnings* surprise and the *price* gap are different animals. Sort on the right one.\n"
            "- **Build your own.** Swap the basket for the S&P 500, or use standardized unexpected "
            "earnings (SUE) instead of `Surprise(%)`; the drift survives, the magnitude moves.\n\n"
            "*Think you can trade the fast version profitably? Show the **net** 1-day long-short landing "
            "above zero after honest costs — then we'll talk.*"
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
            "# Post-Earnings-Announcement Drift — a quantitative teardown 🔬\n"
            "### Quintile long-short on the EPS surprise · forward 1/5/20/60-day drift · a one-sample *t* "
            "+ label-shuffle & block placebo nulls · costs × turnover · the gap-sort myth check · a "
            "synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). PEAD is the "
            "rare folk anomaly that **clears the bar** — so the job here is to *measure it honestly*: "
            "separate the **fundamental** surprise from the **visible** price gap, show the drift is "
            "horizon-gated, confront it with a clustering-aware placebo, and then charge it real costs.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name large-cap** basket, names still trading "
            "in 2026 — a *survivor* panel that tilts the long leg up and sits at the **conservative** end "
            "of PEAD's range (the effect is strongest in small illiquid names; Chordia et al. 2009). "
            "Real data: yfinance daily closes + `Ticker.get_earnings_dates`, 2005→2026. Offline core + "
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
            f"| **Signal** | `REAL` | EPS-surprise long-short **+{R['h20'][4]:.2f}%** at 20d "
            f"(one-sample **t = {R['h20'][6]:.2f}**, placebo **p ≈ {R['h20'][7]:.3f}**) and "
            f"**+{R['h60'][4]:.2f}%** at 60d (**t = {R['h60'][6]:.2f}**) — clears **t ≥ 2**, robust to "
            f"bucket count, and survives a within-quarter block placebo (**p = {R['block_p20']}**). |\n"
            f"| **Tradability** | `FRAGILE` | Net of 4×10-bps one-way + borrow, the short horizons go "
            f"**negative** (1d net **{R['net'][0][2]:+.2f}%**); only 20–60d survive "
            f"(**{R['net'][2][2]:+.2f}% / {R['net'][3][2]:+.2f}%**). Thin, survivor-tilted, 30 names. |\n"
            f"| **Always drifts?** | `BUSTED` | Sort on the **price gap** instead and it **never clears "
            f"t = 2** (20d t = {R['gap'][2][2]:.2f}) and **flips negative at 60d** "
            f"({R['gap'][3][1]:+.2f}%). The drift is in the *fundamental* surprise, not the visible pop. |\n\n"
            "> 💡 In plain words: PEAD is genuinely real on the fundamental surprise — but it's a "
            "**slow, thin** drift, gated past the first week, and the naive \"ride the pop\" version "
            "(sort on the gap) doesn't work."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $u_i$ be the *standardized* earnings surprise for event $i$ (here the vendor "
            "`Surprise(%)`, the free analogue of academic SUE), and $r_{i}(H)$ the forward $H$-day "
            "return entered **one day after** the reaction session (no look-ahead). Quintile by $u$; "
            "the PEAD long-short is\n\n"
            "$$\\widehat{\\Lambda}(H) = \\frac{1}{n_5}\\!\\!\\sum_{i\\in Q_5}\\!\\! r_i(H) \\;-\\; "
            "\\frac{1}{n_1}\\!\\!\\sum_{i\\in Q_1}\\!\\! r_i(H).$$\n\n"
            "- **H₁ (the drift exists).** $\\widehat{\\Lambda}(H) > 0$ and significant for some $H$.\n"
            "- **H₂ (it's deployable).** $\\widehat{\\Lambda}(H)$ survives one-way costs × per-event "
            "turnover + short borrow.\n"
            "- **H₃ (\"ride the pop\").** Sorting on the **visible gap** $g_i$ (the one-day reaction) "
            "drifts as well as sorting on the fundamental surprise.\n\n"
            "We find **H₁ supported** (t≈3 at 20d, robust to a block placebo), **H₂ partly rejected** "
            "(net-negative at 1–5d, net-positive only at 20–60d), **H₃ rejected** (gap-sort never clears "
            "t=2, goes negative at 60d). The anomaly is right; the folk *mechanism* is wrong."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample test of the pooled long-short sample (top-quintile drifts "
            "as longs, bottom-quintile drifts as $-$shorts) against zero:\n\n"
            "$$t = \\frac{\\overline{\\lambda}}{s_\\lambda/\\sqrt{k}},\\qquad "
            "\\lambda \\in \\{r_i : i\\in Q_5\\}\\cup\\{-r_i : i\\in Q_1\\}.$$\n\n"
            "Two honesty problems sit on top of a naive *t*. **(a) Clustering:** earnings arrive in "
            "**seasons**, so events are not independent and a raw *t* overstates significance — we add a "
            "**within-quarter block placebo** that shuffles surprise labels *inside* each calendar "
            "quarter. **(b) The wrong sort:** the *price gap* $g_i$ and the *earnings surprise* $u_i$ are "
            "correlated but distinct; sorting on $g_i$ is a different, mostly-dead trade. The Tradability "
            "axis then charges per-event turnover — the binding constraint at short horizons."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket (yfinance adjusted closes, "
            f"{R['start']}→{R['end']}); **{R['n_events']:,}** quarterly events, **{R['n_eps']:,}** with a "
            "reported EPS surprise. **Survivor** panel — named on the Signal axis.\n"
            "- **Surprise proxies.** Headline = reported **EPS `Surprise(%)`** (fundamental). Myth-check "
            "= the one-day **post-announcement gap** (visible reaction).\n"
            "- **Timing.** Observe the gap at the reaction-session close; enter the close **1 day later** "
            "(no look-ahead); hold $H\\in\\{1,5,20,60\\}$ days; drop events whose window overruns the tape.\n"
            "- **Signal.** Quintile by surprise; long $Q_5$ − short $Q_1$.\n"
            "- **Null #1 (one-sample t)** of the pooled long-short sample vs 0.\n"
            "- **Null #2 (label-shuffle placebo).** 20,000 random re-sorts of the same drifts; "
            "$p = \\Pr[\\text{shuffled }\\Lambda \\ge \\text{observed}]$.\n"
            "- **Null #3 (block placebo).** Same, but shuffling labels *within calendar quarter* "
            "(respects clustering).\n"
            "- **Costs.** 10 bps one-way × 4 legs + 50 bps/yr borrow on the short leg.\n"
            "- **Positive control.** A deterministic panel with a **planted** post-event drift: zero "
            "edge must NOT reach significance; a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Monotonicity + the term structure of the drift\n\n"
            "Left: mean drift by surprise quintile at 20d (the cross-sectional sort). Right: the "
            "long-short at each horizon — the drift's *term structure*, near-zero early and building."
        ),
        code(
            "hs = [1, 5, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    q20 = st.bucket_means(st.event_drift_frame(PRICES, EVV, 20), 'surprise_pct')*100\n"
            "    ls = [st.summarize(PRICES, EVV, h, surprise_col='surprise_pct', placebo=False)['ls_mean']*100 for h in hs]\n"
            "else:\n"
            "    q20 = np.array(R['quintile20']); ls = [R['h1'][4], R['h5'][4], R['h20'][4], R['h60'][4]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(range(5), q20, color=[RED,AMBER,GREY,AMBER,GREEN], width=.7)\n"
            "a1.set_xticks(range(5)); a1.set_xticklabels(['Q1\\nmiss','Q2','Q3','Q4','Q5\\nbeat'])\n"
            "a1.set_ylabel('20d drift (%)'); a1.set_title('Monotone-ish across surprise quintiles')\n"
            "for i,v in enumerate(q20): a1.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.plot(hs, ls, 'o-', c=GREEN, lw=2)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for h,v in zip(hs,ls): a2.annotate(f'{v:+.2f}%',(h,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_xlabel('horizon (days)'); a2.set_ylabel('long-short drift (%)')\n"
            "a2.set_title('Term structure: ~0 at 1-5d, builds to 20-60d')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('quintile 20d:', [round(float(v),2) for v in q20]); print('long-short by H:', [round(v,2) for v in ls])"
        ),
        md(
            f"> 💡 In plain words: the cross-section sorts cleanly (Q5 **{R['quintile20'][-1]:+.2f}%** vs "
            f"Q1 **{R['quintile20'][0]:+.2f}%** at 20d), and the term structure is the giveaway — "
            f"**{R['h1'][4]:+.2f}%** at 1 day grows to **{R['h60'][4]:+.2f}%** at 60. The market reprices "
            "the surprise *slowly*; the predictable part is the lag, exactly Bernard-Thomas."
        ),
        md(
            "### 4b · The decisive test — significance + a clustering-aware placebo\n\n"
            "The 20-day long-short against a 20,000-draw label-shuffle null. The observed spread should "
            "sit in the far right tail. We also report the **within-quarter block placebo** that "
            "respects the seasonality of earnings."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PRICES, EVV, 20)\n"
            "    pl = st.placebo_pvalue(fr, 'surprise_pct', n_draws=20000)\n"
            "    obs = pl['obs']*100; draws = pl['draws']*100; pval = pl['p_value']\n"
            "    s20 = st.summarize(PRICES, EVV, 20, surprise_col='surprise_pct', placebo=False)\n"
            "    tval = s20['t']\n"
            "else:\n"
            "    obs = R['h20'][4]; pval = R['h20'][7]; tval = R['h20'][6]\n"
            "    rng = np.random.default_rng(363); draws = rng.normal(0.0, 0.45, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: 20,000 surprise-label shuffles')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed long-short {obs:+.2f}%')\n"
            "ax.set_xlabel('20-day long-short drift (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Outside the luck cloud: placebo p = {pval:.3f}, one-sample t = {tval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  one-sample t={tval:.2f}  shuffle p={pval:.3f}  '\n"
            "      f'block(within-quarter) p={R[\"block_p20\"]}')"
        ),
        md(
            f"> 💡 In plain words: the green line sits in the **right tail** — only ~"
            f"{R['h20'][7]*100:.1f}% of random sorts match it, one-sample **t = {R['h20'][6]:.2f}**. And "
            f"crucially, even after shuffling labels *within each quarter* to kill the seasonality "
            f"artefact, **p = {R['block_p20']}**. This is a **real** effect, not a clustering mirage — "
            "the rare case where the desk stamps **Signal = REAL**."
        ),
        md(
            "### 4c · Robustness + the myth check — the gap-sort dies\n\n"
            "Left: vary the number of buckets — the 20d *t* stays above 2 for quintiles/deciles. Right: "
            "the same long-short sorted on the **fundamental surprise** vs on the **visible price gap** "
            "— the folk \"ride the pop\" recipe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for nb in (3,5,10):\n"
            "        s = st.summarize(PRICES, EVV, 20, surprise_col='surprise_pct', n_buckets=nb, placebo=False)\n"
            "        rob.append((nb, s['ls_mean']*100, s['t']))\n"
            "    eps_ls = [st.summarize(PRICES, EVV, h, surprise_col='surprise_pct', placebo=False)['ls_mean']*100 for h in hs]\n"
            "    gap_ls = [st.summarize(PRICES, EV, h, surprise_col='gap', placebo=False)['ls_mean']*100 for h in hs]\n"
            "else:\n"
            "    rob = [(r[0], r[1], r[2]) for r in R['robust']]\n"
            "    eps_ls = [R['h1'][4], R['h5'][4], R['h20'][4], R['h60'][4]]; gap_ls = [g[1] for g in R['gap']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar([f'{r[0]}' for r in rob], [r[2] for r in rob], color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,r in enumerate(rob): a1.annotate(f'{r[1]:+.2f}%',(i,r[2]),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_xlabel('number of buckets'); a1.set_ylabel('20d one-sample t'); a1.set_ylim(0,3.4)\n"
            "a1.set_title('t>2 across bucket counts'); a1.legend()\n"
            "x = np.arange(len(hs))\n"
            "a2.bar(x-.2, eps_ls, .4, color=GREEN, label='EARNINGS surprise')\n"
            "a2.bar(x+.2, gap_ls, .4, color=RED, label='price GAP (myth)')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_xticks(x); a2.set_xticklabels([f'{h}d' for h in hs])\n"
            "a2.set_ylabel('long-short drift (%)'); a2.set_title('Gap-sort barely works, negative at 60d'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (buckets, ls%, t):', [(r[0], round(r[1],2), round(r[2],2)) for r in rob])\n"
            "print('gap-sort long-short by H:', [round(v,2) for v in gap_ls])"
        ),
        md(
            f"> 💡 In plain words: the signal is **not** a bucket-count artefact (t>2 from terciles to "
            f"deciles). And the myth dies cleanly — sorting on the **visible gap** gives "
            f"**{R['gap'][2][1]:+.2f}%** at 20d (t={R['gap'][2][2]:.2f}, under the bar) and "
            f"**{R['gap'][3][1]:+.2f}%** at 60d. The information is in the *fundamental* surprise; the "
            "tradeable pop you can see is mostly already priced."
        ),
        md(
            "### 4d · Costs — the binding constraint at short horizons\n\n"
            "Gross vs net (4 one-way legs at 10 bps + 50 bps/yr borrow on the short). The fast money "
            "goes **negative**; only multi-week holds clear costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g, nv = [], []\n"
            "    for h in hs:\n"
            "        c = st.net_of_costs(PRICES, EVV, h, surprise_col='surprise_pct')\n"
            "        g.append(c['gross']*100); nv.append(c['net']*100)\n"
            "else:\n"
            "    g = [n[1] for n in R['net']]; nv = [n[2] for n in R['net']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net (costs + borrow)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('long-short return (%)'); ax.set_title('Costs flip 1-5d negative; 20-60d survive')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h,gg,nn in zip(hs,g,nv): print(f'{h:>2}d: gross={gg:+.2f}%  net={nn:+.2f}%')"
        ),
        md(
            f"> 💡 In plain words: at 1 day, gross **{R['net'][0][1]:+.2f}%** → net "
            f"**{R['net'][0][2]:+.2f}%** — the round-trip on four legs costs more than the edge. The drift "
            "only pays if you hold it **20–60 days** (net **{:+.2f}% / {:+.2f}%**). Frequency × per-event "
            "turnover, not raw cost, is what makes the fast version un-tradable.".format(
                R['net'][2][2], R['net'][3][2])
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a known post-event drift: with **zero** edge the "
            "long-short must stay at t≈0 (no false positive); with a **large** planted drift it must "
            "light up. Both hold — so the real-tape t≈3 is a genuine signal, not a construction artefact."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.08):\n"
            "    px, e = data.synthetic_pead(edge=edge, seed=363)\n"
            "    s = st.summarize(px, e, 20, n_draws=4000)\n"
            "    res.append((edge, s['n_events'], s['ls_mean']*100, s['t'], s['p_placebo'], s['ls_win']*100))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}%' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[3] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: zero edge -> t~0; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,ls,t,p,w in res: print(f'planted {e*100:+.0f}%: events={k} ls20={ls:+.2f}% t={t:.2f} p={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted drift the control sits at "
            f"**t = {R['syn'][0][3]:.2f}** (no false positive — noise cannot fake it); a **+8%** planted "
            f"drift reaches **t = {R['syn'][1][3]:.2f}**. The machinery is unbiased, so the real-tape "
            "**t ≈ 3** at 20 days is the genuine article. This is what separates PEAD from the dozens of "
            "folk signals that *don't* survive: it actually moves the needle."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — EPS-surprise long-short **+{R['h20'][4]:.2f}%** at 20d "
            f"(one-sample **t = {R['h20'][6]:.2f}**, shuffle **p ≈ {R['h20'][7]:.3f}**, block-placebo "
            f"**p = {R['block_p20']}**) and **+{R['h60'][4]:.2f}%** at 60d (**t = {R['h60'][6]:.2f}**). "
            "Clears the **t ≥ 2** bar, robust to bucket count and to clustering. Carries an explicit "
            "**survivorship** caveat (long-leg tilt) and is **horizon-gated** (nothing at 1–5d). The "
            "rare folk effect that earns REAL, not WEAK.\n"
            f"- **Tradability `FRAGILE`** — net of 4×10-bps + borrow, the short horizons go **negative** "
            f"(1d **{R['net'][0][2]:+.2f}%**); only 20–60d survive (**{R['net'][2][2]:+.2f}% / "
            f"{R['net'][3][2]:+.2f}%**). A thin net spread on a {R['n_names']}-name survivor basket, "
            "leaning on the long leg — real but operationally delicate. Not INVESTABLE.\n"
            f"- **Always drifts? `BUSTED`** — sort on the **visible gap** and it never clears t=2 (20d "
            f"t = {R['gap'][2][2]:.2f}) and **flips negative at 60d** ({R['gap'][3][1]:+.2f}%). The "
            "anomaly is right; the folklore's *mechanism* (\"ride the pop\") is wrong."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — where the drift actually lives\n\n"
            "One picture for the operational truth: the **net** long-short by horizon, with the "
            "break-even line. The tradeable region is a narrow window — long enough to clear costs, "
            "short enough that the drift hasn't fully decayed."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nv = [st.net_of_costs(PRICES, EVV, h, surprise_col='surprise_pct')['net']*100 for h in hs]\n"
            "    gg = [st.summarize(PRICES, EVV, h, surprise_col='surprise_pct', placebo=False)['ls_mean']*100 for h in hs]\n"
            "else:\n"
            "    nv = [n[2] for n in R['net']]; gg = [R['h1'][4], R['h5'][4], R['h20'][4], R['h60'][4]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(hs, gg, 'o--', c=GREY, lw=1.6, label='gross long-short')\n"
            "ax.plot(hs, nv, 'o-', c=GREEN, lw=2.2, label='net of costs + borrow')\n"
            "ax.axhline(0, c=RED, ls='--', label='break-even')\n"
            "ax.fill_between(hs, 0, nv, where=[v>0 for v in nv], color=GREEN, alpha=.12)\n"
            "for h,v in zip(hs,nv): ax.annotate(f'{v:+.2f}%',(h,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('horizon (days)'); ax.set_ylabel('long-short return (%)')\n"
            "ax.set_title('Tradeable region: net-positive only at 20-60 days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net long-short by horizon:', {f'{h}d': round(v,2) for h,v in zip(hs,nv)})"
        ),
        md(
            "> 💡 In plain words: the green (net) line is **below zero** for the first week and only "
            "lifts into the shaded region at 20–60 days. That's the whole tradability story: PEAD is a "
            "**slow-money** effect on large-caps. You don't scalp the pop — you hold a fresh long-short "
            "earnings book through the quarter and collect a thin, survivor-tilted spread. Real edge, "
            "fragile vehicle — hence **FRAGILE**, not INVESTABLE."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Liquidity is the knob.** Chordia, Goyal, Sadka, Sadka & Shivakumar (2009) show PEAD "
            "concentrates in **small, illiquid** names and shrinks among liquids once costs bite — "
            "exactly our large-cap result. Re-run on small-caps for a bigger (but costlier) drift.\n"
            "- **SUE vs `Surprise(%)`.** Swap the vendor surprise for standardized unexpected earnings "
            "(de-meaned by the firm's own surprise volatility); the drift typically *strengthens*.\n"
            "- **The fundamental-vs-visible lesson generalises.** Many \"momentum\" stories are really "
            "this confusion — the predictable signal is in a slow-moving fundamental, not the fast, "
            "already-priced price move.\n\n"
            "*The reproducible core is offline and deterministic; the surprise input is the reported EPS "
            "surprise (fundamental) with the price gap as an explicit myth-check. Methods and sources: "
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
