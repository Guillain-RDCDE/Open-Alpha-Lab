"""Generate the two narrative notebooks for Study 316 (Bank-Failure).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic. The real-tape cells use the cached daily
parquet under ../_cache/ if present (HAVE_REAL) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader,
online or off. Every real-tape cell banners which tape it is showing.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-18).
R = dict(
    spy_fp="c3d89ac3d3f5", xlf_fp="78429d2d69ba",
    spy_days=8404, xlf_days=6914, n_events=12, n_fit=11,
    # SPY day-0 and CARs
    spy_day0=-1.53, spy_car5=-1.35, spy_car5_med=-0.02, spy_car5_win=45, spy_car5_t=-0.71,
    spy_car20=-5.50, spy_car20_med=3.20, spy_car20_win=64, spy_car20_t=-0.93,
    spy_car20_lo=-17.28, spy_car20_hi=3.95, spy_car60=-5.8, spy_car60_t=-0.77,
    # cluster split
    c2008_n=5, c2008_mean=-15.5, c2023_n=4, c2023_mean=4.6,
    rest_n=6, rest_mean=2.8, rest_med=5.1, rest_t=1.34,
    drop_worst=-2.4,
    # XLF
    xlf_day0=-3.23, xlf_car20=-9.11, xlf_car20_win=36, xlf_car20_t=-1.24,
    xlf_rest_mean=-1.7, xlf_rest_med=-2.3,
    # placebo confound
    plac_z=-4.49, plac_p=0.0001, plac_mean=0.81, xlf_plac_z=-4.63,
    # overlay
    ov0=-2.79, ov0_win=67, ov0_t=-0.62, ov5=-2.89, ov10=-2.99, ov10_t=-0.67, ov_n=12,
    lag1_mean=-3.03, lag1_z=-2.74,
)

# Per-event SPY +20d CARs (real, as-of 2026-06-18) — for the punchline bar chart.
PER_EVENT = [
    ("Northern Rock '07", 5.3), ("Bear Stearns '08", 3.3), ("IndyMac '08", 3.2),
    ("Lehman '08", -21.3), ("WaMu '08", -26.0), ("Wachovia '08", -36.4),
    ("MF Global '11", -6.9), ("SVB '23", 4.9), ("Signature '23", 6.4),
    ("Credit Suisse '23", 6.0), ("First Republic '23", 1.0),
]


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from bank_failure import data, strategy as st

EV = data.events(as_of="2026-06-18")

def _have_real():
    try:
        return (os.path.exists(data._cache_path("SPY", data.DEFAULT_CACHE)))
    except Exception:
        return False

HAVE_REAL = _have_real()
TAPE = "REAL tape (Yahoo daily, adjusted, as-of 2026-06-18)" if HAVE_REAL else "OFFLINE — frozen numbers from docs/results.md"
print("real daily cache present:", HAVE_REAL, "|", TAPE)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Bank-Failure — blood in the streets, or the first domino? 🏦\n"
            "### When SVB, Lehman or Credit Suisse go down, is the market a buy or a warning?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_reliable_warning%3F: Misattributed](https://img.shields.io/badge/A_reliable_warning%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "Every time a big bank fails, two camps shout past each other. The contrarians "
            "quote Baron Rothschild — *'buy when there's blood in the streets'* — and Warren "
            "Buffett's 2008 *Buy American* op-ed: the panic *is* the bottom. The doom camp says "
            "a bank failure is the **first visible domino** — Lehman, then the whole 2008 "
            "collapse. This notebook lines up the actual market move after a dozen headline "
            "failures and asks who's right.\n\n"
            "> **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "capacity math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is a bank failure a buy signal? | **No clear signal either way.** Over the 20 days "
            f"after, SPY's *average* is **{R['spy_car20']:.1f}%** — but the *median* is "
            f"**+{R['spy_car20_med']:.1f}%** and most failures were followed by an *up* market. |\n"
            "| Is it a reliable warning of a crash? | **No — it's a hindsight illusion.** The whole "
            f"scary average is **one cluster: autumn 2008** (avg {R['c2008_mean']:.0f}%). Strip 2008 "
            f"and the other failures averaged **+{R['rest_mean']:.1f}%**. |\n"
            "| Could you trade 'buy the blood'? | **No.** Buying SPY on each failure and holding 20 "
            f"days **loses {R['ov0']:.1f}% per trade** before costs. |\n"
            "| So what is the lesson? | A sample of **eleven** clustered events can't certify "
            "anything. 'Bank failure' is a label pasted onto 'it was 2008' — and 2023 (SVB, Credit "
            f"Suisse) actually **bounced +{R['c2023_mean']:.1f}%**. |\n\n"
            "> The honest answer is the boring one: a headline bank failure tells you almost nothing "
            "about where the market goes next. The story only *feels* like a warning because the "
            "worst failures all happened in the same terrible autumn."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Buy when there is blood in the streets, even if the blood is your own.'* "
            "— attributed to Baron Rothschild\n\n"
            "> *'Be fearful when others are greedy, and greedy when others are fearful... "
            "I've been buying American stocks.'* — Warren Buffett, *Buy American. I Am.* "
            "(New York Times, October 2008)\n\n"
            "Against them, the contagion camp (Reinhart & Rogoff, *This Time Is Different*): a "
            "bank failure is rarely a one-off — it's the leading edge of a credit crunch. **Lehman** "
            "is their exhibit A: it failed on 15 September 2008, and the market fell off a cliff for "
            "months. So which is it — *buy the panic*, or *run for the exits*?"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This is the kind of question where your *intuition* is built almost entirely from "
            "**one** memory — 2008 — and that's exactly the trap. If 'bank failure = sell' were "
            "true, you'd have a rare, high-conviction macro timing rule. If 'bank failure = buy' "
            "were true, you'd have a contrarian playbook for the scariest days. And if **neither** "
            "is true — if the average is just one disaster dressed up as a rule — then the real "
            "lesson is about how a tiny, clustered sample fools us into seeing a signal. That "
            "lesson transfers to every 'X always precedes a crash' story you'll ever hear."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We do the obvious thing: make a list of the **public dates** big banks failed or were "
            "force-sold (the day the market knew), then look at what SPY did over the next 20 "
            "trading days after each one. Three honesty checks:\n\n"
            "1. **Look at every event, not just the average.** A mean can be one disaster; the "
            "median and the per-event chart can't hide.\n"
            "2. **Split 2008 from everything else.** If the 'warning' is really just the 2008 "
            "cluster, that split will scream it.\n"
            "3. **Race it against random dates.** Run the same window on random calendar days — if "
            "post-failure windows look like any other window, there's no signal.\n\n"
            "Tape: SPY daily bars back to 1993 (and XLF, the financials sector, in the quant "
            "notebook)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the punchline in one chart: what did SPY do over the 20 days after *each* "
            "failure?**"
        ),
        code(
            "labels = [e[0] for e in " + repr(PER_EVENT) + "]\n"
            "vals = [e[1] for e in " + repr(PER_EVENT) + "]\n"
            "if HAVE_REAL:\n"
            "    spy = data.load_real('SPY', fetch=False)\n"
            "    win = st.event_window_returns(spy, EV['date'], pre=10, post=20)\n"
            "    car = (st.post_event_car(win, 20) * 100)\n"
            "    labels = [d.strftime('%b %Y') for d in win.index]\n"
            "    vals = list(car)\n"
            "fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "col = [GREEN if v >= 0 else RED for v in vals]\n"
            "ax.bar(range(len(vals)), vals, color=col)\n"
            "ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('SPY return, +20 trading days (%)')\n"
            "ax.set_title('Most bank failures were followed by an UP market — except autumn 2008')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Banner:', TAPE)\n"
            "print('up after 20d:', sum(v>0 for v in vals), 'of', len(vals), 'events')"
        ),
        md(
            f"Look at the colours. **Most bars are green** — the market was *higher* 20 days after "
            f"the failure. The three monsters (Lehman, WaMu, Wachovia, all **autumn 2008**) are the "
            f"only thing dragging the average underwater. The 2023 cluster — SVB, Signature, Credit "
            f"Suisse, First Republic — is **all green**."
        ),
        md(
            "**So is the average a 'warning'? Watch what happens when we separate 2008 from "
            "everything else:**"
        ),
        code(
            "groups = ['2008 cluster', '2023 cluster', 'Everything\\nexcept 2008']\n"
            "if HAVE_REAL:\n"
            "    c = pd.Series(st.post_event_car(win, 20)*100, index=win.index)\n"
            "    g2008 = c[(c.index>='2008-01-01')&(c.index<'2009-01-01')].mean()\n"
            "    g2023 = c[c.index>='2023-01-01'].mean()\n"
            "    grest = c[(c.index<'2008-01-01')|(c.index>='2009-01-01')].mean()\n"
            "    means = [g2008, g2023, grest]\n"
            "else:\n"
            f"    means = [{R['c2008_mean']}, {R['c2023_mean']}, {R['rest_mean']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "b = ax.bar(groups, means, color=[RED, GREEN, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean SPY +20d return (%)')\n"
            "ax.set_title('The bearish \\'warning\\' is just one cluster — autumn 2008')\n"
            "for bar_, m in zip(b, means):\n"
            "    ax.annotate(f'{m:+.1f}%', (bar_.get_x()+bar_.get_width()/2, m),\n"
            "        ha='center', va='bottom' if m>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Banner:', TAPE)"
        ),
        md(
            f"There it is. **2008 averaged {R['c2008_mean']:.0f}%**; **2023 averaged "
            f"+{R['c2023_mean']:.1f}%**; everything that wasn't 2008 averaged **+{R['rest_mean']:.1f}%** "
            f"(a +{R['rest_med']:.1f}% median). The 'bank failures warn of crashes' rule is really "
            "'the worst banking crash in a century was a crash.' Drop the single worst day and the "
            f"average already halves (to {R['drop_worst']:.1f}%)."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** SPY's average 20-day move is {R['spy_car20']:.1f}% but the median "
            f"is *positive* (+{R['spy_car20_med']:.1f}%) and on eleven clustered events the statistics "
            "can't tell it apart from noise.\n"
            "- **Tradability — Mirage.** 'Buy the blood' (buy SPY on each failure, hold 20 days) "
            f"**loses {R['ov0']:.1f}% per trade** before any cost. Twelve trades in forty years — "
            "there's nothing to scale.\n"
            "- **A reliable warning? — Misattributed.** The bearish average is the autumn-2008 "
            f"cluster; strip it and failures averaged **+{R['rest_mean']:.1f}%**. The warning is "
            "hindsight: 'bank failure' is a label for 'it was 2008.'"
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Suppose you believed the contrarians and bought SPY on every failure, holding 20 days. "
            "Here's the per-trade result, and why costs aren't even the problem:"
        ),
        code(
            "costs = [0.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    spy = data.load_real('SPY', fetch=False)\n"
            "    net = [st.summarize_trades(st.buy_the_blood(spy, EV['date'], 20, cost_bps=c),'ret_net')['mean_pct'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['ov0']}, {R['ov5']}, {R['ov10']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar([str(c) for c in costs], net, color=RED, width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('one-way cost (bps)')\n"
            "ax.set_ylabel('mean return per trade (%)')\n"
            "ax.set_title('\\'Buy the blood\\' loses money before a cent of cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Banner:', TAPE)"
        ),
        md(
            f"The overlay is **already negative at zero cost** ({R['ov0']:.1f}%/trade) — costs barely "
            f"register because the gross is the problem. It even *wins* {R['ov0_win']:.0f}% of the "
            "time, which is the oldest trap in the book: a high win-rate with a negative average means "
            "many small rebounds and a few catastrophic 2008 trades. (The desk has a whole study on "
            "that illusion: [Study 301 — Triple-RSI](../../301-triple-rsi/).)"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The generic cousin.** [Study 241 — Buy-the-Dip](../../241-buy-the-dip/) asks the "
            "same question about *any* dip (also None / Mirage). Bank-Failure just conditions on a "
            "scarier-sounding trigger and lands in the same place.\n"
            "- **Other 'this signals a crash' omens.** [Study 167 — Hindenburg-Omen](../../167-hindenburg-omen/) "
            "and [Study 160 — Skyscraper-Curse](../../160-skyscraper-curse/) — same small-sample, "
            "hindsight-clustered failure mode.\n"
            "- **The honest open question.** With only ~12 events ever, *no* statistical test can "
            "settle this. The right takeaway isn't 'buy' or 'sell' — it's that a clustered handful of "
            "events is un-certifiable, and your gut is just replaying 2008.\n\n"
            "*Think the 2023 bounce proves 'buy the blood'? Fork this, add the next failure when it "
            "comes, and watch the average swing on a single event — that's the whole point.*"
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
            "# Bank-Failure — a quantitative teardown 🔬\n"
            "### Event study · CAR + HAC *t* · block-bootstrap CI · placebo/permutation control · "
            "the clustering confound · a planted-drift positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_reliable_warning%3F: Misattributed](https://img.shields.io/badge/A_reliable_warning%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We run a classic daily event "
            "study around a hardcoded table of bank-failure dates, on SPY and the financials sector "
            "(XLF), and confront the one thing that makes this question almost un-answerable: with "
            "eleven clustered events, the statistics are weak by construction.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars (adjusted), SPY back to 1993, "
            "XLF back to 1998; as-of 2026-06-18; the offline core and tests run on a deterministic "
            "synthetic tape. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | SPY +20d mean CAR **{R['spy_car20']:.2f}%**, HAC/plain "
            f"*t* = **{R['spy_car20_t']:.2f}** (n={R['n_fit']}); 95% block-bootstrap CI "
            f"**[{R['spy_car20_lo']:.1f}%, +{R['spy_car20_hi']:.1f}%]**; median **+{R['spy_car20_med']:.1f}%**. |\n"
            f"| **Tradability** | `MIRAGE` | 'Buy the blood' overlay **{R['ov0']:.2f}%/trade gross** "
            f"(*t* = {R['ov0_t']:.2f}); negative before cost; n={R['ov_n']} trades. |\n"
            f"| **A reliable warning?** | `MISATTRIBUTED` | 2008 cluster {R['c2008_mean']:.1f}% vs "
            f"ex-2008 **+{R['rest_mean']:.1f}%** (median +{R['rest_med']:.1f}%); 2023 cluster "
            f"+{R['c2023_mean']:.1f}%. |\n\n"
            "> 💡 In plain words: the bearish average is one cluster (autumn 2008), the *t*-stat is "
            "nowhere near the bar, and the only 'significant' number — a placebo p-value — is "
            "measuring *'was it 2008?'*, not *'is a bank failure a signal?'*"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For event $e$ on its public date, let $\\mathrm{CAR}_e(h)=\\sum_{k=0}^{h} r_{e,k}$ be the "
            "cumulative log return from the event-day close to $+h$ trading days. Across the event "
            "set $E$:\n\n"
            "$$\\overline{\\mathrm{CAR}}(h)=\\frac{1}{|E|}\\sum_{e\\in E}\\mathrm{CAR}_e(h)$$\n\n"
            "- **H₁ (warning).** $\\overline{\\mathrm{CAR}}(20) < 0$ with $|t|\\ge 2$ — failures "
            "systematically precede declines.\n"
            "- **H₂ (buy the blood).** $\\overline{\\mathrm{CAR}}(20) > 0$ with $|t|\\ge 2$ — failures "
            "are buyable panics.\n"
            "- **H₃ (it's just 2008).** The sign of $\\overline{\\mathrm{CAR}}$ flips once the "
            "autumn-2008 cluster is removed.\n"
            "- **H₄ (tradable).** A long-SPY overlay entered on each event has positive net "
            "expectancy.\n\n"
            "We find **H₁ and H₂ both rejected** ($|t|<2$), **H₃ supported** (sign flips), **H₄ "
            "rejected** (negative gross)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A genuine $|t|\\ge 2$ either way would be a rare, actionable macro-event signal. The "
            "more important content here is methodological: this is a textbook case of **event "
            "clustering** (MacKinlay 1997) defeating naive inference. The five worst events share one "
            "calendar window, so cross-sectional independence — the assumption behind the standard "
            "event-study *t* — is violated, and a random-date placebo silently asks the wrong "
            "question. Naming that precisely is the deliverable."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Events.** Public-knowledge failure/rescue dates (announcement / FDIC seizure / "
            "forced sale). Calendar-known → **no execution lag** at the anchor (a `lag=1` next-close "
            "variant is the robustness check).\n"
            "- **Window.** Log returns on $[-10, +20]$ trading days; CAR cumulated from day 0.\n"
            "- **Inference.** HAC/plain *t* on the per-event CAR (tiny n → plain *t*); circular "
            "block-bootstrap CI (preserves the 2008/2023 clustering); a permutation/placebo test "
            "against random calendar dates — reported but **explicitly not used to stamp**.\n"
            "- **Cluster split.** 2008 vs 2023 vs ex-2008.\n"
            "- **Tradability.** Long-SPY overlay, +20d hold, one-way cost × NAV on both legs.\n"
            "- **Positive control.** A deterministic synthetic tape with a *planted* post-event drift "
            "— the harness must recover it and call the null a null.\n\n"
            "Tickers: SPY (1993–), XLF (1998–)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The CAR curve and its honest *t*\n\n"
            "The mean cumulative path around the events, with the per-event CARs and HAC/plain *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = data.load_real('SPY', fetch=False)\n"
            "    win = st.event_window_returns(spy, EV['date'], pre=10, post=20)\n"
            "    curve = st.car_curve(win) * 100\n"
            "    car20 = st.post_event_car(win, 20)\n"
            "    t20 = st.hac_tstat(car20); lo, hi = st.block_bootstrap_ci(car20, seed=316)\n"
            "    mean20, med20 = car20.mean()*100, np.median(car20)*100\n"
            "else:\n"
            "    rel = list(range(-10, 21))\n"
            f"    mean20, med20, t20, lo, hi = {R['spy_car20']}, {R['spy_car20_med']}, {R['spy_car20_t']}, {R['spy_car20_lo']/100}, {R['spy_car20_hi']/100}\n"
            "    # a stylised path consistent with the frozen endpoints\n"
            "    curve = pd.Series(np.interp(rel, [-10,0,20], [1.0, -1.53, mean20]), index=rel)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(curve.index, curve.values, lw=2, color=RED)\n"
            "ax.axvline(0, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_betweenx([curve.min(), curve.max()], 0, 20, color=GREY, alpha=.08)\n"
            "ax.set_xlabel('trading days relative to the failure'); ax.set_ylabel('mean cumulative return (%)')\n"
            "ax.set_title(f'SPY around bank failures: mean {mean20:+.1f}% at +20d (median {med20:+.1f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Banner:', TAPE)\n"
            "print(f'+20d: mean {mean20:+.2f}%  median {med20:+.2f}%  t={t20:+.2f}  95% CI [{lo*100:+.1f}%, {hi*100:+.1f}%]')"
        ),
        md(
            f"> 💡 In plain words: the mean dips to **{R['spy_car20']:.1f}%** but the *t* is "
            f"**{R['spy_car20_t']:.2f}** and the 95% CI **[{R['spy_car20_lo']:.1f}%, "
            f"+{R['spy_car20_hi']:.1f}%]** straddles zero by a mile. With the median *positive* "
            f"(+{R['spy_car20_med']:.1f}%), the mean is a few disasters, not a drift. **H₁ and H₂ both "
            "fail the bar.**"
        ),
        md(
            "### 4b · It's the 2008 cluster — the misattribution\n\n"
            "Split the events and the 'warning' evaporates."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = pd.Series(st.post_event_car(win, 20)*100, index=win.index)\n"
            "    g08 = c[(c.index>='2008-01-01')&(c.index<'2009-01-01')]\n"
            "    g23 = c[c.index>='2023-01-01']\n"
            "    rest = c[(c.index<'2008-01-01')|(c.index>='2009-01-01')]\n"
            "    rows = [('2008 cluster', len(g08), g08.mean()),\n"
            "            ('2023 cluster', len(g23), g23.mean()),\n"
            "            ('ex-2008', len(rest), rest.mean())]\n"
            "    rest_t = st.hac_tstat(rest.values)\n"
            "else:\n"
            f"    rows = [('2008 cluster', {R['c2008_n']}, {R['c2008_mean']}),\n"
            f"            ('2023 cluster', {R['c2023_n']}, {R['c2023_mean']}),\n"
            f"            ('ex-2008', {R['rest_n']}, {R['rest_mean']})]\n"
            f"    rest_t = {R['rest_t']}\n"
            "dfc = pd.DataFrame(rows, columns=['subset','n','mean_car20_%'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(dfc['subset'], dfc['mean_car20_%'], color=[RED, GREEN, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean +20d CAR (%)')\n"
            "ax.set_title('Sign flips once you remove autumn 2008 — the warning is hindsight')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Banner:', TAPE); print(dfc.round(2).to_string(index=False))\n"
            "print(f'ex-2008 t = {rest_t:+.2f} (still below the bar — even the bounce is uncertifiable)')"
        ),
        md(
            f"> 💡 In plain words: **2008 = {R['c2008_mean']:.1f}%**, **ex-2008 = +{R['rest_mean']:.1f}%** "
            f"(median +{R['rest_med']:.1f}%), **2023 = +{R['c2023_mean']:.1f}%**. The sign of the whole "
            "result is decided by one cluster. **H₃ supported.** (Note: even ex-2008 the *t* is "
            f"+{R['rest_t']:.2f} — too few events to certify the *bounce* either. Nobody gets a stamp.)"
        ),
        md(
            "### 4c · Why the placebo p-value is a trap\n\n"
            "A naive permutation test (event CAR vs random-date placebos) looks 'significant'. We "
            "show it — and then explain why it does **not** earn a stamp."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.permutation_pvalue(spy, EV['date'], n_placebo=2000, horizon=20, seed=316)\n"
            "    ev_car, pmu, psd, z, p = res['event_car'], res['placebo_mean'], res['placebo_std'], res['z'], res['p_value']\n"
            "    rng = np.random.default_rng(0)\n"
            "    draws = []\n"
            "    for _ in range(2000):\n"
            "        rd = data.random_dates(spy, n=len(win), margin=30, seed=int(rng.integers(0,2**31-1)))\n"
            "        pw = st.event_window_returns(spy, rd, pre=10, post=20)\n"
            "        pc = st.post_event_car(pw, 20)\n"
            "        if pc.size: draws.append(pc.mean()*100)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            f"    ev_car, pmu, z, p = {R['spy_car20']/100}, {R['plac_mean']/100}, {R['plac_z']}, {R['plac_p']}\n"
            "    rng = np.random.default_rng(0); draws = rng.normal(0.8, 1.4, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.7, label='random-date placebos')\n"
            "ax.axvline(ev_car*100, color=RED, lw=2, label=f'events ({ev_car*100:+.1f}%)')\n"
            "ax.set_xlabel('mean +20d CAR (%)'); ax.set_ylabel('placebo count')\n"
            "ax.set_title(f\"'Significant' placebo p={p:.3f}, z={z:+.1f} — but it's a confound\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Banner:', TAPE)\n"
            "print(f'event {ev_car*100:+.2f}%  placebo mean {pmu*100:+.2f}%  z={z:+.2f}  p={p:.4f}')"
        ),
        md(
            f"> 💡 In plain words: the events sit far in the left tail of the random-date cloud "
            f"(z = {R['plac_z']:.1f}, p < 0.001) — but the events **all cluster inside one bear "
            "market**, so this test is really detecting *'were these dates in 2008?'*, not *'is a "
            "bank failure a signal?'* Per the desk's inference bar, a synthetic/placebo control is a "
            "**machinery proof, never market evidence**. The per-event *t* "
            f"({R['spy_car20_t']:.2f}) is the number that decides, and it says **None**."
        ),
        md(
            "### 4d · The afflicted sector — XLF (financials)\n\n"
            "Financials fall harder (it's their own sector imploding), but the *t* is still below 2 "
            "and ex-2008 the move is modest."
        ),
        code(
            "if HAVE_REAL and os.path.exists(data._cache_path('XLF', data.DEFAULT_CACHE)):\n"
            "    xlf = data.load_real('XLF', fetch=False)\n"
            "    wx = st.event_window_returns(xlf, EV['date'], pre=10, post=20)\n"
            "    cx = st.post_event_car(wx, 20)\n"
            "    xlf_m, xlf_t, xlf_w = cx.mean()*100, st.hac_tstat(cx), (cx>0).mean()*100\n"
            "else:\n"
            f"    xlf_m, xlf_t, xlf_w = {R['xlf_car20']}, {R['xlf_car20_t']}, {R['xlf_car20_win']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.0))\n"
            f"ax.bar(['SPY','XLF (financials)'], [{R['spy_car20']}, xlf_m], color=[AMBER, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean +20d CAR (%)')\n"
            "ax.set_title(f'XLF falls harder ({xlf_m:+.1f}%, t={xlf_t:+.2f}) — still uncertifiable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Banner:', TAPE); print(f'XLF +20d: {xlf_m:+.2f}%  t={xlf_t:+.2f}  win {xlf_w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: XLF averages **{R['xlf_car20']:.1f}%** (*t* = {R['xlf_car20_t']:.2f}), "
            f"and ex-2008 only **{R['xlf_rest_mean']:.1f}%**. Even the directly-hit sector carries no "
            "certifiable post-failure signal once the cluster is removed."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — SPY +20d mean {R['spy_car20']:.2f}%, HAC/plain *t* "
            f"{R['spy_car20_t']:.2f} (n={R['n_fit']}); 95% CI [{R['spy_car20_lo']:.1f}%, "
            f"+{R['spy_car20_hi']:.1f}%]; median +{R['spy_car20_med']:.1f}%. XLF *t* {R['xlf_car20_t']:.2f}. "
            "Indistinguishable from noise.\n"
            f"- **Tradability `MIRAGE`** — overlay {R['ov0']:.2f}%/trade gross (*t* {R['ov0_t']:.2f}), "
            f"negative before cost, n={R['ov_n']}. Nothing to scale.\n"
            f"- **A reliable warning? `MISATTRIBUTED`** — 2008 {R['c2008_mean']:.1f}% vs ex-2008 "
            f"+{R['rest_mean']:.1f}%; 2023 +{R['c2023_mean']:.1f}%. The sign is one cluster."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the overlay, gross and net\n\n"
            "The 'buy the blood' overlay, charged a one-way cost × NAV on both legs."
        ),
        code(
            "costs = [0.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    led0 = st.buy_the_blood(spy, EV['date'], 20, cost_bps=0.0)\n"
            "    s0 = st.summarize_trades(led0, 'ret_gross')\n"
            "    net = [st.summarize_trades(st.buy_the_blood(spy, EV['date'], 20, cost_bps=c),'ret_net')['mean_pct'] for c in costs]\n"
            "    win_rate, ov_t = s0['win_rate']*100, s0['tstat']\n"
            "else:\n"
            f"    net = [{R['ov0']}, {R['ov5']}, {R['ov10']}]; win_rate, ov_t = {R['ov0_win']}, {R['ov0_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar([str(c) for c in costs], net, color=RED, width=.5)\n"
            "a1.axhline(0, c='k', lw=1); a1.set_xlabel('one-way cost (bps)'); a1.set_ylabel('mean %/trade')\n"
            "a1.set_title('Negative before any cost')\n"
            "a2.bar(['win rate','mean/trade'], [win_rate, net[0]], color=[GREY, RED], width=.5)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_title(f'{win_rate:.0f}% wins, yet {net[0]:+.1f}% mean')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Banner:', TAPE); print(f'gross {net[0]:+.2f}%/trade  t={ov_t:+.2f}  win {win_rate:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: {R['ov0_win']:.0f}% of trades 'win', yet the mean is "
            f"**{R['ov0']:.1f}%** — the win-rate-vs-expectancy illusion (a few 2008 trades sink it). "
            "Costs are a rounding error because the gross is already a loss. **MIRAGE.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always cry 'noise'? Plant a known "
            "post-event drift and sweep it — the harness must find it when it's there and ignore it "
            "when it isn't."
        ),
        code(
            "drifts = [-0.06, 0.0, 0.04, 0.08, 0.12]\n"
            "zs, ps = [], []\n"
            "for dr in drifts:\n"
            "    c, ev, _ = data.synthetic_tape(n_days=6000, drift=dr, seed=316)\n"
            "    res = st.permutation_pvalue(c, ev['date'], n_placebo=300, horizon=20, seed=1)\n"
            "    zs.append(res['z']); ps.append(res['p_value'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if p < 0.05 else GREY for p in ps]\n"
            "ax.bar([f'{d:+.2f}' for d in drifts], zs, color=col)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted post-event drift'); ax.set_ylabel('event-vs-placebo z')\n"
            "ax.set_title('The engine finds a planted drift and calls the null a null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Banner: SYNTHETIC positive control (always offline)')\n"
            "print('z by drift:', [round(z,2) for z in zs])\n"
            "print('p by drift:', [round(p,3) for p in ps])"
        ),
        md(
            "The engine's z is ~0 at drift 0 (it correctly finds nothing) and grows with the planted "
            "drift (it correctly finds something) — so the real-tape `NONE` is a statement about "
            "**the market**, not a broken pipeline. The market simply hasn't produced enough "
            "bank-failure events, spread across enough independent regimes, to certify anything.\n\n"
            "For the generic version of this question see "
            "[Study 241 — Buy-the-Dip](../../241-buy-the-dip/); for the win-rate illusion that sinks "
            "the overlay, [Study 301 — Triple-RSI](../../301-triple-rsi/)."
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
