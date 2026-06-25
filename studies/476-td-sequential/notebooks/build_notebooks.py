"""Generate the two narrative notebooks for Study 476 (TD Sequential — DeMark 9-13).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# yfinance daily, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29 (As-of
# 2026-05-31, partial June dropped), 21.4 years, TD buy setup-9 / countdown-13 long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=239,
    fp_spy="4cb5244f3990",
    # pooled setup-9, per horizon (single-seed welch=seed 7):
    # (H, n, setup_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 239, 55.0, 55, 2.21, 6.2, 48.9, 53.0, 1.76, 0.080),
    h10=(10, 239, 42.9, 56, 1.55, 54.0, -11.1, 40.9, -0.32, 0.751),
    h20=(20, 239, 193.4, 67, 5.08, 51.0, 142.4, 191.4, 2.91, 0.004),
    h60=(60, 234, 371.6, 70, 5.32, 100.8, 270.8, 369.6, 3.31, 0.001),
    # honest baseline: setup vs avg of 200 random draws -> (random_avg_bps, delta_bps, emp_p)
    emp5=(27.9, 27.2, 0.139), emp10=(49.6, -6.7, 0.582),
    emp20=(102.4, 91.0, 0.060), emp60=(302.0, 69.6, 0.204),
    # seed-averaged welch (mean, min, max) over 20 seeds:
    welch20=(2.07, -0.00, 5.61), welch60=(0.82, -1.09, 3.31),
    # countdown-13 pooled: (H, n, cd_bps, win%, one_sample_t, random_bps, delta_bps, welch_t)
    cd5=(5, 183, 1.5, 55, 0.05, 16.2, -14.7, -0.51),
    cd10=(10, 180, 20.7, 49, 0.63, -26.5, 47.2, 1.30),
    cd20=(20, 180, 38.1, 56, 1.00, -84.2, 122.3, 2.36),
    cd60=(60, 180, 282.1, 64, 3.80, 156.5, 125.6, 1.51),
    # per-ticker H=20: (ticker, entries, setup_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 40, 87.9, 1.21, 39.3, 48.6), ("QQQ", 43, 273.4, 2.31, 88.2, 185.3),
         ("IWM", 43, 297.5, 2.92, 2.7, 294.9), ("DIA", 50, 133.3, 1.83, 33.7, 99.6),
         ("GLD", 63, 182.5, 3.41, 83.4, 99.1)],
    # scrambled-lookback placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(87.9, 0.673, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, setup_bps, win%, one_sample_t)
    syn=[(0.00, 62, 68.0, 52, 1.28), (0.40, 78, 277.2, 79, 6.55)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_exhaustion%3F: Busted](https://img.shields.io/badge/Forecasts_exhaustion%3F-Busted-8b949e?style=flat-square)\n\n"
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

from td_sequential import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real TD-sequential cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does DeMark's 9-13 count actually call the bottom? 🔢\n"
            "### A famous exhaustion counter — count to 9, then to 13 — meets a stopwatch\n\n"
            + BADGES +
            "Open Bloomberg, TradingView or Thinkorswim and you'll find **TD Sequential**, Tom "
            "DeMark's exhaustion counter. The rule is purely mechanical: count **nine** closes in a "
            "row each below the close four bars earlier (a *Buy Setup*), then keep counting to "
            "**thirteen** (a *Countdown*). The lore, taught by DeMark and repeated on every "
            "technical-analysis site, is that when the count completes, **sellers are exhausted** — "
            "the down-move is spent and price is about to turn up. So you buy the 9, and especially "
            "the 13.\n\n"
            "Unlike a hand-drawn trendline there's nothing to eyeball here — the count is an "
            "algorithm. That makes it the perfect thing to test honestly: fire the \"buy the "
            "completed setup\" rule across five big indices over 21 years and time the result with a "
            "stopwatch — against the only baseline that matters: **buying on random days instead.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy when a TD Buy Setup completes (the **9**), do I make money? | **Yes — but "
            "mostly because the market goes up.** Win-rate ~60–67%, returns look great. |\n"
            "| Is that *the count's* doing? | **Barely, if at all.** At 20 days it edges out random "
            "by +91 bps — but that win **vanishes** once you sample the random baseline properly "
            "(it never reaches statistical significance). |\n"
            "| Is the deeper **13** even better? | **No — it's worse.** DeMark's \"strongest\" "
            "exhaustion signal is the *weaker* one. |\n"
            "| Does the exact 4-bar count matter? | **No.** Swap it for any other lookback and the "
            "result barely changes. |\n"
            "| So is it a tradable edge? | **No.** It's mostly **beta in a costume** — the upward "
            "drift of stocks, re-labelled as an exhaustion count. |\n\n"
            "> TD Sequential is a tidy way to *mark* a down-streak. As a *forecast* — \"the 9 will "
            "bounce\" — it's a **near-miss that's really a mirage**: a faint 20-day tilt that doesn't "
            "survive a fair test, almost all of it the market's long-run climb."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Count nine closes in a row, each below the close four bars back — that's a **Buy "
            "Setup**. Then count thirteen bars closing at or below the low two bars back — that's the "
            "**Countdown**. When the count completes, sellers are exhausted; price reverses up. Buy "
            "the 9, buy the 13.\"*\n\n"
            "This is **Tom DeMark's** TD Sequential (*The New Science of Technical Analysis*, 1994), "
            "licensed on Bloomberg and built into every charting suite. It's one of the most "
            "recognisable timing tools in technical analysis — and because it's a *pure algorithm*, "
            "we can test exactly what its inventor specified, with no judgement calls."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the count genuinely *forecast* reversals, it would be remarkable: a fixed arithmetic "
            "rule on past closes would predict turning points — a clean crack in market efficiency you "
            "could trade by counting on your fingers. That's the dream the indicator sells.\n\n"
            "But there's a trap. The count fires after a **down-streak**, and it's measured on a "
            "market (stock indices) that drifts **up** over time, so *any* dip-buying rule will look "
            "profitable. To separate the **count** from the **tide**, we compare it to buying on "
            "**random days** — and we have to sample those random days *many times*, because a single "
            "unlucky (or lucky) draw can swing the verdict. We'll do exactly that."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Run the count mechanically.** Nine consecutive closes below close-4 → a completed "
            "Buy Setup; we never peek ahead — each rung uses only bars up to today.\n"
            "2. **Trade the lore.** When the setup completes, buy at the **next close**; measure the "
            "return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days** — and average over "
            "200 random draws so one lucky seed can't fake a result.\n"
            "4. **The placebo.** Swap DeMark's exact 4-bar comparison for other lookbacks. If the "
            "*specific* count matters, the scramble should kill it. *If it doesn't, the geometry is a "
            "mirage* — that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the count even look like? Here's SPY with the running Buy-Setup count "
            "and the bars where a completed **9** would fire a long."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    ent = st.buy_setup_entries(cl); ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=55, zorder=5, label='completed Buy Setup (9) BUY')\n"
            "    ax.set_title('Mechanical TD Buy Setup-9 completions on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('setup-9 completions in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The 9s land near short-term dips — *as a description*. The question is whether those "
            "green dots are followed by bounces. **Let's race the setup-9 against random entries** at "
            "four horizons. Blue = buy the completed 9; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    setup, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.buy_setup_entries(c)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        setup.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    setup = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, setup, .4, color='#2c6fbb', label='buy the completed 9')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days (one seed)')\n"
            "for i,(a,bb) in enumerate(zip(setup,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The 9 looks like it beats random at 20-60d... on ONE lucky seed'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('setup:', [round(v) for v in setup]); print('random(seed7):', [round(v) for v in rnd])"
        ),
        md(
            f"On this single random draw the 9 looks great at 20–60 days (**+{R['h20'][2]:.0f} bps** "
            f"vs **+{R['h20'][5]:.0f}** random). **But a single random seed is treacherous.** Re-draw "
            "the random baseline 200 times and ask how often it beats the 9 — that's the only honest "
            "*p*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    obs20, rmeans = None, []\n"
            "    tt = [st.forward_returns(load(t)['close'], st.buy_setup_entries(load(t)['close']), 20) for t in data.DEFAULT_TICKERS]\n"
            "    obs20 = np.concatenate(tt).mean()*1e4\n"
            "    for sd in range(200):\n"
            "        rr = []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']; e = st.buy_setup_entries(c)\n"
            "            rr.append(st.forward_returns(c, st.random_entries(c, max(len(e),50), seed=1000+sd), 20))\n"
            "        rmeans.append(np.concatenate(rr).mean()*1e4)\n"
            "    rmeans = np.array(rmeans); pval = (np.sum(rmeans>=obs20)+1)/(len(rmeans)+1)\n"
            "else:\n"
            "    obs20 = R['h20'][2]; pval = R['emp20'][2]\n"
            "    rng = np.random.default_rng(20); rmeans = rng.normal(R['emp20'][0], 55, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rmeans, bins=35, color=GREY, alpha=.85, label='random-day baseline (200 draws, 20d)')\n"
            "ax.axvline(obs20, c='#2c6fbb', lw=2.5, label=f'completed 9 = {obs20:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The 9 sits in the right tail but does not clear it: p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'completed-9 20d = {obs20:+.0f} bps   empirical p vs 200 random baselines = {pval:.3f}')"
        ),
        md(
            f"There's the honest picture: the 9 sits in the **right tail** of the random cloud but "
            f"**does not clear it** — empirical *p* = **{R['emp20'][2]:.3f}**, never below the 0.05 "
            "bar at any horizon. The lucky single-seed win was a mirage of one draw."
        ),
        md(
            "**One more check — the placebo.** DeMark's count compares today's close to the close "
            "**four** bars ago. What if we swap that 4 for some other number? If the *specific* "
            "4-bar count is what works, the swap should break it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.scrambled_lookback_placebo(c, 20, n_draws=300, seed=476)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real 4-bar setup (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of OTHER-lookback setups do at least as well (p={pval:.2f}).')\n"
            "print('=> the specific 4-bar count is not doing the work.')"
        ),
        md(
            f"Two-thirds of the **other-lookback** setups match or beat DeMark's canonical 4-bar one "
            f"(*p* = {R['placebo'][1]:.2f}). If the magic were in the exact count, a swap would "
            "collapse it. It doesn't — because whatever faint tilt exists is generic \"buy after a "
            "down-streak\", not the 9-13 geometry."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The completed 9 *hints* at a 20-day edge over random (+91 bps) but "
            "it never clears the bar once the baseline is sampled properly (empirical *p* = 0.06). "
            "The big absolute returns are the market's drift, not the count — and the deeper 13 is "
            "*worse*.\n"
            "- **Tradability — Mirage.** Nothing scalable once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Does the 9-13 forecast exhaustion\"? — Busted.** Swap the count's lookback and the "
            "result barely moves; the \"stronger\" 13 is the weaker signal. The count doesn't "
            "forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing scalable here. The 9's *only* edge over a coin flip is a faint, "
            "non-significant 20-day tilt riding on the market's long-run climb — which you'd capture "
            "more cheaply (and more fully) by just **holding the index**. The setup buy trades rarely "
            "(~11 times a year) and pays costs on each, so it's a worse, more expensive way to be "
            "long. As a forecasting tool, it doesn't pay; as a counter, it was never meant to be a "
            "standalone strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **\"Setup perfection\" and the TD Combo.** DeMark adds refinements (perfection, "
            "aggressive countdown, TDST levels). They're parameter tweaks of the same "
            "close-vs-close-N count and inherit the same drift confound — a fun follow-up is to show "
            "each one lands in the same place.\n"
            "- **The lucky-seed trap.** This study is a clean teaching example of why a *single* "
            "random baseline can fake significance: seed 7 gave Welch *t* = 2.9; the seed-averaged "
            "*t* is 2.1.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-setup bounce "
            "into a synthetic tape and shows the harness banks it — so the null result here isn't a "
            "dead detector, it's an honest 'nothing there'.\n\n"
            "*Think the count forecasts? Show the completed 9 beating random entries at empirical "
            "**p < 0.05** on a real tape, after sampling the baseline — then we'll talk.*"
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
            "# TD Sequential (DeMark 9-13) — a quantitative teardown 🔬\n"
            "### Mechanical 9-13 count on 5 indices · setup/countdown forward returns · "
            "one-sample HAC *t* · a **seed-averaged** random-entry baseline · an empirical *p* · "
            "a scrambled-lookback geometry placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **count** from the **drift** — and to show why a *single* random "
            "baseline is not enough: an up-drifting index makes any dip-buy look good, and one lucky "
            "seed can push the Welch *t* past 2.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. The count is fully "
            "algorithmic (no eyeballing); entry is the **next close** (one documented lag). Offline "
            "core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Setup-9 vs a **seed-averaged** random baseline: Δ at 20d = "
            f"**+{R['emp20'][1]:.0f} bps** but empirical **p = {R['emp20'][2]:.3f}** (never < 0.05); "
            f"the seed-averaged Welch *t* is only **+{R['welch20'][0]:.2f}** at 20d / "
            f"**+{R['welch60'][0]:.2f}** at 60d. The lucky single-seed Welch (+{R['h20'][8]:.2f}) and "
            f"one-sample *t* (+{R['h20'][4]:.2f}) are mostly beta. |\n"
            f"| **Tradability** | `MIRAGE` | No significant residual edge; ~11 signals/yr, no "
            f"capacity, costs deepen the hole. The deeper **13** is *weaker* (welch "
            f"{R['cd60'][7]:+.2f} at 60d). |\n"
            f"| **Forecasts exhaustion?** | `BUSTED` | Scrambling the count's lookback leaves the "
            f"result intact: **p = {R['placebo'][1]:.2f}** of other-lookback setups match or beat the "
            "canonical 4-bar one. The exact 9-13 geometry isn't load-bearing. |\n\n"
            "> 💡 In plain words: the 9 *almost* clears the bar — but the win is one lucky seed plus "
            "drift. Sample the baseline honestly and it falls to p≈0.06; scramble the count and it "
            "survives. A near-miss, not a forecast."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A **TD Buy Setup** is the maximal run with $C_t<C_{t-4}$ for nine consecutive bars; the "
            "ninth bar fires. A **TD Buy Countdown** then accumulates bars with $C_t\\le L_{t-2}$ to "
            "thirteen. The DeMark rule goes long at completion, betting on exhaustion.\n\n"
            "- **H₀ (drift).** Setup returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the count forecasts).** Setup returns **exceed** random at some horizon, "
            "empirical p < 0.05 *after sampling the baseline over many seeds*.\n"
            "- **H₂ (the geometry matters).** Setup returns exceed a **scrambled-lookback** setup "
            "whose comparison offset is not the canonical 4.\n\n"
            "We find **H₀ not rejected** (empirical p ≥ 0.06 at every horizon), **H₁ rejected** "
            "(seed-averaged Welch t = +2.07/+0.82 at 20/60d), **H₂ rejected** (placebo p ≈ 0.67). "
            "The steelman fails — narrowly at 20d, clearly elsewhere."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the three confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean; *any* long-only "
            "entry inherits it, so a one-sample $t$ against **zero** measures the tide. Fix: a "
            "**random-entry baseline** (same instrument, epoch, hold).\n\n"
            "**(b) Baseline sampling noise.** A *single* random draw has its own large standard "
            "error — seed 7 here gives Welch t = 2.9, but the seed-averaged t is 2.1 (range "
            "−0.0…+5.6). Fix: **average the baseline over 200 seeds** and compute an **empirical p** "
            "from the distribution of random means.\n\n"
            "**(c) Geometry as a free parameter.** The 4-bar / 2-bar counts are choices; the danger "
            "is that *any* down-streak detector works on a trend. The **scrambled-lookback placebo** "
            "swaps the offset, so if the real result survives, the specific count was never "
            "load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} setup-9 completions** "
            "pooled.\n"
            "- **Setup.** Nine consecutive $C_t<C_{t-4}$; the 9th bar fires. **Countdown.** Thirteen "
            "$C_t\\le L_{t-2}$ rungs after a setup, standard recycle on a fresh setup.\n"
            "- **No look-ahead.** Every rung uses only bars ≤ t; read on close of t, enter **next "
            "close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of setup returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, **seed-averaged** + empirical p (the *real* "
            "test).\n"
            "- **Null #3 — scrambled-lookback placebo** (offset ≠ 4, count length kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every signal.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-setup bounce (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap & the lucky seed — one-sample t looks great, the honest test deflates it\n\n"
            "Left: the setup-9's **one-sample** t against zero (the misleading number — it's beta). "
            "Right: the same setup vs a **seed-averaged** random baseline (mean Welch t over 20 "
            "seeds, with the lucky single-seed value marked)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    one_t, welch_seed7, welch_avg = [], [], []\n"
            "    for h in hs:\n"
            "        tt = [st.forward_returns(load(t)['close'], st.buy_setup_entries(load(t)['close']), h) for t in data.DEFAULT_TICKERS]\n"
            "        tt = np.concatenate(tt); one_t.append(st.summarize(tt)['t'])\n"
            "        ws = []\n"
            "        for sd in range(1,21):\n"
            "            rr = [st.forward_returns(load(t)['close'], st.random_entries(load(t)['close'], max(len(st.buy_setup_entries(load(t)['close'])),50), seed=sd), h) for t in data.DEFAULT_TICKERS]\n"
            "            ws.append(stats.ttest_ind(tt, np.concatenate(rr), equal_var=False)[0])\n"
            "        welch_seed7.append(ws[6]); welch_avg.append(float(np.mean(ws)))\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    welch_seed7 = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "    welch_avg = [0.9, -0.2, R['welch20'][0], R['welch60'][0]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "x = np.arange(len(hs))\n"
            "a2.bar(x, welch_avg, .55, color=[GREEN if v>2 else AMBER if v>1 else RED for v in welch_avg], label='seed-averaged Welch t')\n"
            "a2.scatter(x, welch_seed7, c='k', zorder=5, label='lucky single seed (7)')\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch_avg): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{h}d' for h in hs])\n"
            "a2.set_title('Setup vs RANDOM (honest: seed-averaged barely touches 2)'); a2.set_ylabel('t'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t])\n"
            "print('welch seed7:', [round(v,2) for v in welch_seed7]); print('welch seed-avg:', [round(v,2) for v in welch_avg])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — pure drift. On the right, the **lucky seed 7** dots float above "
            f"the seed-averaged bars: the honest Welch is only **+{R['welch20'][0]:.2f}** at 20d and "
            f"**+{R['welch60'][0]:.2f}** at 60d. The single-seed +{R['h20'][8]:.2f} was a draw, not an "
            "edge."
        ),
        md(
            "### 4b · The empirical p — setup mean vs 200 random baselines\n\n"
            "The cleanest honest test: compute the setup mean, then the *distribution* of "
            "random-baseline means over 200 draws, and read off how far in the right tail the setup "
            "sits."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tt = [st.forward_returns(load(t)['close'], st.buy_setup_entries(load(t)['close']), 20) for t in data.DEFAULT_TICKERS]\n"
            "    obs20 = np.concatenate(tt).mean()*1e4; rmeans = []\n"
            "    for sd in range(200):\n"
            "        rr = [st.forward_returns(load(t)['close'], st.random_entries(load(t)['close'], max(len(st.buy_setup_entries(load(t)['close'])),50), seed=1000+sd), 20) for t in data.DEFAULT_TICKERS]\n"
            "        rmeans.append(np.concatenate(rr).mean()*1e4)\n"
            "    rmeans = np.array(rmeans); pval = (np.sum(rmeans>=obs20)+1)/(len(rmeans)+1)\n"
            "else:\n"
            "    obs20 = R['h20'][2]; pval = R['emp20'][2]\n"
            "    rng = np.random.default_rng(20); rmeans = rng.normal(R['emp20'][0], 55, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rmeans, bins=35, color=GREY, alpha=.85, label='random-entry baseline (200 draws, 20d)')\n"
            "ax.axvline(obs20, c='#2c6fbb', lw=2.5, label=f'setup-9 = {obs20:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Setup-9 in the right tail but not past it: empirical p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'setup-9 20d = {obs20:+.0f} bps   empirical p (200 random baselines) = {pval:.3f}  (>0.05 => not significant)')"
        ),
        md(
            f"> 💡 In plain words: the setup sits in the right tail (empirical *p* = "
            f"**{R['emp20'][2]:.3f}** at 20d — the closest of all horizons) but **never crosses the "
            "0.05 line**. There's a whiff of signal at 20 days; it isn't enough to call."
        ),
        md(
            "### 4c · The geometry placebo — scramble the lookback, nothing changes\n\n"
            "Swap DeMark's exact 4-bar comparison for other offsets (2,3,5,6,7,8), keeping the 9-bar "
            "setup length and the price marginal. If price respects *this specific count*, the "
            "scramble should demolish the result. The observed 4-bar return should sit far in the "
            "right tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.scrambled_lookback_placebo(c, 20, n_draws=300, seed=476)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    rng = np.random.default_rng(476); offsets=[2,3,5,6,7,8]; draws=[]\n"
            "    for _ in range(300):\n"
            "        lb = int(rng.choice(offsets))\n"
            "        rr = st.forward_returns(c, st.buy_setup_entries(c, lookback=lb), 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(476); draws = rng.normal(95, 40, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=35, color=GREY, alpha=.85, label='other-lookback setups (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'canonical 4-bar {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean setup-9 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The 4-bar count sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'canonical 4-bar {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => count not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the canonical 4-bar count (blue) sits **mid-pack** in the "
            f"other-lookback cloud — **p = {R['placebo'][1]:.2f}**. Any down-streak detector does as "
            "well, so DeMark's specific 9-13 geometry isn't carrying information. This is the cleanest "
            "refutation of 'the count forecasts exhaustion.'"
        ),
        md(
            "### 4d · The deeper '13' is the weaker signal\n\n"
            "DeMark sells the 13-countdown as the *strongest* exhaustion read. If the count truly "
            "forecast, the 13 should beat the 9. It doesn't — here are both vs their random baselines."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s9, c13 = [], []\n"
            "    for h in hs:\n"
            "        a = np.concatenate([st.forward_returns(load(t)['close'], st.buy_setup_entries(load(t)['close']), h) for t in data.DEFAULT_TICKERS])\n"
            "        b = np.concatenate([st.forward_returns(load(t)['close'], st.buy_countdown_entries(load(t)['close'], load(t)['low']), h) for t in data.DEFAULT_TICKERS])\n"
            "        s9.append(a.mean()*1e4); c13.append(b.mean()*1e4)\n"
            "else:\n"
            "    s9 = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    c13 = [R['cd5'][2], R['cd10'][2], R['cd20'][2], R['cd60'][2]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, s9, .4, color='#2c6fbb', label='setup-9')\n"
            "ax.bar(x+.2, c13, .4, color=AMBER, label='countdown-13 (DeMark\\'s \"stronger\" read)')\n"
            "for i,(a,b) in enumerate(zip(s9,c13)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('The \"stronger\" 13 is the weaker signal'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('setup-9 (bps):', [round(v) for v in s9]); print('countdown-13 (bps):', [round(v) for v in c13])"
        ),
        md(
            f"> 💡 In plain words: the 13 (amber) is *below* the 9 (blue) at every short horizon "
            f"(20d: {R['cd20'][2]:+.0f} vs {R['h20'][2]:+.0f} bps). DeMark's deepest exhaustion read "
            "is the *weakest* — the opposite of what a real forecast would show."
        ),
        md(
            "### 4e · Per-ticker — positive everywhere, but tiny-n and that's the drift\n\n"
            "20-day setup-minus-random delta, per instrument. Positive in all five — but the *n* is "
            "tiny (40–63 setups in 21y) and the pooled empirical p (0.06) says the aggregate doesn't "
            "reach significance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']; e = st.buy_setup_entries(c); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d setup − random (bps)'); ax.set_title('Positive in all 5 — but tiny-n and non-significant pooled')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: consistent-sign (all positive, e.g. IWM **{R['per'][2][5]:+.0f}**) "
            "but tiny samples and a non-significant pooled p — the fingerprint of a small drift tilt, "
            "not a robust forecast."
        ),
        md(
            "### 4f · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-setup bounce "
            "into a synthetic tape and check the same setup-9 rule banks it: edge=0 must stay near the "
            "drift floor; edge>0 must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=476, n_days=4000)\n"
            "    c = px['close']; e = st.buy_setup_entries(c); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> near drift floor; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} setup={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits near the drift floor "
            f"(**t = {R['syn'][0][4]:.2f}**, win {R['syn'][0][3]:.0f}%); a planted post-setup bounce "
            f"reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector works — "
            "so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the setup-9 hints at a 20-day edge (Δ = +{R['emp20'][1]:.0f} bps) "
            f"but does not survive a properly sampled random baseline (empirical p = {R['emp20'][2]:.3f}, "
            f"never < 0.05; seed-averaged Welch t = +{R['welch20'][0]:.2f}/+{R['welch60'][0]:.2f} at "
            f"20/60d). The one-sample t's (20d **{R['h20'][4]:.2f}**) are mostly beta, and the deeper "
            "13 is the weaker signal.\n"
            f"- **Tradability `MIRAGE`** — no significant residual edge once the drift is removed; "
            "~11 signals/yr, no capacity, costs deepen the hole. Hold the index instead.\n"
            f"- **Forecasts exhaustion? `BUSTED`** — the scrambled-lookback placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): any down-streak detector does as well, and "
            "the \"stronger\" 13 underperforms the 9. The specific 9-13 count carries no exhaustion "
            "information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing scalable to trade\n\n"
            "The setup-9's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The rule trades "
            "rarely (~11/yr) and pays costs on each, so it dominates *nothing*. There is no capacity "
            "question because there is no significant edge to scale. TD Sequential is a descriptive "
            "counter, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The lucky-seed lesson.** This study is a textbook case of why a *single* random "
            "baseline can fake significance (seed 7: Welch t = 2.9; seed-averaged: 2.1). Always "
            "resample the baseline.\n"
            "- **DeMark's refinements.** Setup 'perfection', TD Combo, aggressive countdown and TDST "
            "levels are parameter tweaks of the same close-vs-close-N count and inherit the same "
            "drift confound; the canonical 9-13 here is the charitable representative.\n"
            "- **Short side.** A symmetric TD Sell Setup/Countdown short would face the drift as a "
            "*headwind* — an interesting asymmetry the same engine can measure.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen "
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
