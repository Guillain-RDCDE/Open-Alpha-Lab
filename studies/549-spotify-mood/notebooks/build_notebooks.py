"""Generate the two narrative notebooks for Study 549 (Spotify-Mood).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic cells run
anywhere, offline and deterministic; the real-tape cells use the cached monthly ^GSPC under
../_cache/ if present and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader offline. The MOOD series is synthetic in
every cell — never under a real-tape banner (only the *market* is real).
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; real ^GSPC 2010-02..2026-05,
# 196 months; mkt fp 431d740a90aa; synthetic null mood; joined panel fp 1e087c2e441e).
R = dict(
    fp_mkt="431d740a90aa", fp_panel="1e087c2e441e",
    n=196, n_reg=195, start="2010-02", end="2026-05",
    slope=-0.0032, ols_t=-1.06, hac_t=-1.25, r2=0.006,
    placebo_p=0.231,
    hit=46.7, base=66.2,
    # lag sweep: (lag, slope, hac_t)
    sweep=[(1, -0.003, -1.25), (2, -0.005, -2.11), (3, -0.003, -1.16),
           (4, -0.004, -1.77), (5, -0.004, -1.73)],
    bonf=2.58,
    # trading
    lo_gross=4.7, lo_net=4.6, ls_gross=-3.3, ls_net=-4.0, mkt_ann=13.2,
    cost_bps=5.0, borrow_bps=100.0,
    # synthetic control: (predictive_beta, mean HAC t over 25 seeds)
    ctrl=[(0.000, -0.03), (0.005, 1.54), (0.010, 3.11), (0.015, 4.68), (0.020, 6.25)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from spotify_mood import data, strategy as st

def load_real():
    \"\"\"Cache-first REAL market join with the SYNTHETIC null mood proxy (empty offline).\"\"\"
    ret = data.fetch_market(fetch=False)
    if ret is None or len(ret) == 0:
        return pd.DataFrame()
    return data.join_synthetic_valence(ret)

FRAME = load_real()
HAVE_REAL = not FRAME.empty
print("real market tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(FRAME)} months, {FRAME.index[0].date()} -> {FRAME.index[-1].date()})")
print("NOTE: the MOOD series is SYNTHETIC in every cell (no real valence tape exists);"
      " only the MARKET is real.")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Spotify-Mood — do happier songs foretell a happier market? 🎧\n"
            "### Can the *mood of what we stream* predict next month's stocks, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Music_sentiment_on_the_tape%3F: Untestable](https://img.shields.io/badge/Music_sentiment_on_the_tape%3F-Untestable-8b949e?style=flat-square)\n\n"
            "Here's a fun idea from finance research. Spotify tags every song with a number called "
            "**valence** — roughly, how *happy* it sounds (a bouncy pop hit scores high; a mournful "
            "ballad scores low). A 2022 study found that when a country's most-streamed songs get "
            "*happier*, its stock market tends to be doing better too. So maybe the collective mood "
            "in our headphones **leads** the market — sadder charts today, weaker stocks tomorrow?\n\n"
            "We'd love to test that on real data. **We can't** — and *why* we can't is half the "
            "story. So we do the honest next-best thing and check whether a plausible, well-behaved "
            "mood series predicts the real market at all.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool. The **market** data is "
            "real; the **mood** data is synthetic (see below). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can we even get the real 'song mood' data? | **No.** Spotify shut off the feature that "
            "exposes song *valence* to new apps in **Nov 2024**, and never published a clean history "
            "of chart mood. The famous study bought a **private** dataset. So our mood tape has to be "
            "**made up** (carefully, and seeded). |\n"
            f"| Does a plausible mood series predict next month's market? | **No.** The link is the "
            f"*wrong direction* and tiny (HAC *t* = {R['hac_t']}), and a shuffle test says it's noise "
            f"(*p* = {R['placebo_p']}). |\n"
            f"| Does it call the market's direction? | **Worse than a coin.** {R['hit']}% right, when "
            f"just always guessing 'up' would be right {R['base']}% of the time. |\n"
            f"| Could you trade it? | **No.** The mood-timing rule made **+{R['lo_gross']}%/yr** while "
            f"the market did **+{R['mkt_ann']}%/yr** — it sits out the good months. |\n\n"
            "> The academic result is real *on private data*. On a free retail stack the claim is "
            "**untestable** — so we instead prove our measuring machine works (it catches a fake edge "
            "we plant on purpose), then show a plausible real-world mood series carries no signal."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The average happiness (valence) of the top-streamed songs is a mood proxy that moves "
            "with — and leads — the stock market.\"* — the music-sentiment literature (Edmans, "
            "Fernandez-Perez, Garel & Indriawan 2022).\n\n"
            "The intuition: markets are partly driven by *feelings*. When people feel good, they buy "
            "risky assets; when they feel gloomy, they sell. And what we stream is a giant, honest "
            "thermometer of the collective mood. If mood really leads prices, tracking song valence "
            "would be an early-warning gauge for the market. It sits in the same family as *sunshine* "
            "moving stocks, or a national football loss denting the market the next day."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it's a genuinely new, hard-to-arbitrage signal — nobody can trade *your* "
            "playlist. The desk has already poked at other mood proxies: "
            "[256 Twitter-Mood](../../256-twitter-mood/) (social-media mood) and "
            "[300 Sports-Sentiment](../../300-sports-sentiment/) (football results). This one is the "
            "**music** proxy. But first we hit a wall that decides everything: **can we get the "
            "data?**"
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "The clean test is simple: line up each month's average song valence next to the market's "
            "return *the following month*, and see if happy months are followed by strong markets.\n\n"
            "The problem is the very first step. **Spotify closed the door.** The only public source "
            "of a song's valence — the 'audio-features' API — stopped accepting new apps in November "
            "2024, and it never offered a tidy monthly history of chart mood anyway. The researchers "
            "who found the effect *licensed a private dataset*. A hobbyist can't reach it.\n\n"
            "So we do two honest things instead:\n"
            "1. **Prove the machine works.** Build a *fake* mood series where we secretly plant a real "
            "predictive edge, and check our test catches it. (It does — see beat 7.)\n"
            "2. **Test a plausible real-world mood series** — a realistic, drifting mood index with "
            "*no* secret edge — against the **real** market, and see if it accidentally predicts "
            "anything. (It shouldn't.)\n\n"
            "*Timing:* the rule only ever uses *last* month's completed mood to call *this* month — "
            "no peeking at the future (one honest execution lag)."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Does last month's mood predict this month's market?** Each dot is a month: mood on the "
            "x-axis, the *next* month's market return on the y-axis. A real 'happy leads up' signal "
            "would tilt the cloud upward-to-the-right."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.predictive_regression(FRAME, lag=1)\n"
            "    v = st._z(FRAME['valence'].to_numpy())[:-1]\n"
            "    y = FRAME['mkt_ret'].to_numpy()[1:]*100\n"
            "    slope, hac_t = reg['slope'], reg['hac_t']\n"
            f"    banner = 'REAL ^GSPC market ({R['n']} months) x SYNTHETIC mood proxy'\n"
            "else:\n"
            f"    slope, hac_t = {R['slope']}, {R['hac_t']}\n"
            "    rng = np.random.default_rng(549)\n"
            "    v = rng.standard_normal(180); y = 0.6 + 4.5*rng.standard_normal(180)\n"
            "    banner = 'frozen numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(v, y, s=26, color=GREY, alpha=.8)\n"
            "xs = np.linspace(v.min(), v.max(), 50)\n"
            "ax.plot(xs, (slope*100)*xs + y.mean(), c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('this month mood (valence, z-scored)'); ax.set_ylabel('NEXT month market return %')\n"
            "ax.set_title(f'Almost flat, and tilted the WRONG way (HAC t {hac_t:+.2f})')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'slope {slope:+.4f} per +1 sigma mood -> next-month return; HAC t {hac_t:+.2f}')"
        ),
        md(
            f"There's essentially **nothing**, and what little there is points the *wrong way* (a "
            f"happier month is followed by a very slightly *weaker* market — HAC *t* {R['hac_t']}, well "
            "inside noise). The claim wanted an upward tilt; the cloud is a shapeless blob."
        ),
        md(
            "**Is it just bad luck at one lag?** Maybe mood leads by 2 or 3 months instead of 1. Let's "
            "try several — but note the trap: test enough lags and *one* will look good by chance."
        ),
        code(
            "sweep = " + repr([(s[0], s[2]) for s in R["sweep"]]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.lag_sweep(FRAME)\n"
            "    lags = list(tab.index); ts = list(tab['hac_t'])\n"
            "else:\n"
            "    lags = [s[0] for s in sweep]; ts = [s[1] for s in sweep]\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "cols = [RED if abs(t)>=2 else GREY for t in ts]\n"
            "ax.bar([str(l) for l in lags], ts, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1); ax.axhline(-2, ls='--', c=GREEN, lw=1)\n"
            f"ax.axhline({R['bonf']}, ls=':', c='k', lw=1); ax.axhline({-R['bonf']}, ls=':', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('lag (months mood leads)'); ax.set_ylabel('HAC t')\n"
            "ax.set_title('Test 5 lags, one grazes -2 (lag 2) - but wrong sign & below the Bonferroni bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HAC t by lag:', [round(t,2) for t in ts])"
        ),
        md(
            f"One lag (2 months) grazes *t* = −2 — but it's *negative* (the wrong sign for 'happy → "
            f"up'), it's below the stricter bar you need when you test 5 things at once (~{R['bonf']}), "
            "and the other four lags say nothing. That's textbook data-mining, not a signal."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** No real mood data exists to test the claim, and a plausible mood "
            f"series shows a wrong-signed, insignificant link (HAC *t* {R['hac_t']}, placebo *p* "
            f"{R['placebo_p']}).\n"
            f"- **Tradability — Mirage.** The mood-timing rule (+{R['lo_gross']}%/yr) badly trails "
            f"just holding the market (+{R['mkt_ann']}%/yr).\n"
            "- **Music sentiment on the tape? — Untestable.** Real on a *licensed* dataset; "
            "unreachable on a free stack — so we verify the *engine*, not the claim."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Suppose you tried anyway: hold the market in happy-mood months, sit in cash otherwise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lo = st.mood_timing(FRAME, long_short=False)\n"
            "    g, mk = lo['gross_ann']*100, lo['mkt_ann']*100\n"
            "else:\n"
            f"    g, mk = {R['lo_gross']}, {R['mkt_ann']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['mood-timing\\nrule','just hold\\nthe market'], [g, mk], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annual return %')\n"
            "ax.set_title(f'Timing on mood: +{g:.0f}%/yr vs simply holding +{mk:.0f}%/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mood-timing +{g:.1f}%/yr  vs  buy-and-hold +{mk:.1f}%/yr')"
        ),
        md(
            "The rule *underperforms simply owning the market* by roughly 8 points a year — it keeps "
            "guessing wrong about when to be in, and misses the rallies. `MIRAGE`."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Does our test even work?** Yes — in the quant notebook we *plant* a real mood edge "
            "and watch the test light up (past *t* = +2). So the flat reading here is the data "
            "talking, not a broken tool.\n"
            "- **Other mood proxies.** [256 Twitter-Mood](../../256-twitter-mood/) and "
            "[300 Sports-Sentiment](../../300-sports-sentiment/) test the same 'collective mood moves "
            "markets' idea with different thermometers.\n\n"
            "*Got a real, survivorship-clean monthly valence tape (licensed data, an old archive)? "
            "Fork this, drop it into `data.py` in place of the synthetic proxy, and see if the HAC *t* "
            "clears +2 the right way.*"
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
            "# Spotify-Mood — a quantitative teardown 🔬\n"
            "### Synthetic valence proxy × real ^GSPC · Newey-West HAC predictive regression · lag-sweep / Bonferroni trap · circular-shift placebo · hit-rate vs base rate · gross/net mood-timing · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Music_sentiment_on_the_tape%3F: Untestable](https://img.shields.io/badge/Music_sentiment_on_the_tape%3F-Untestable-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the aggregate "
            "musical **valence** of top-streamed songs leads the market — and confront the hard "
            "constraint that decides the Signal axis: **no free real valence tape exists.**\n\n"
            "> ⚠️ **Not investment advice.** The **market** is real (monthly ^GSPC, 2010-02 → 2026-05, "
            f"fp `{R['fp_mkt']}`, joined panel fp `{R['fp_panel']}`); the **mood** series is "
            "**synthetic** in *every* cell (Spotify closed the audio-features API to new apps in Nov "
            "2024 — see [`docs/references.md`](../docs/references.md)). Numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | No real valence tape (synthetic-only → capped). Honest test: "
            f"lag-1 slope **{R['slope']}** (wrong sign), HAC *t* **{R['hac_t']}**, placebo *p* "
            f"**{R['placebo_p']}**, hit-rate **{R['hit']}%** vs base **{R['base']}%**. |\n"
            f"| **Tradability** | `MIRAGE` | Long-only mood-timing **+{R['lo_gross']}%/yr** vs market "
            f"**+{R['mkt_ann']}%/yr**; long-short **{R['ls_gross']}%** gross → **{R['ls_net']}%** net "
            f"({R['cost_bps']:.0f} bps/leg + {R['borrow_bps']:.0f} bps borrow). |\n"
            "| **Music sentiment on the tape?** | `UNTESTABLE` | Real on a *licensed* dataset (Edmans "
            "et al. 2022); unreachable free — engine verified on synthetic worlds instead. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but there is no "
            "real valence tape to run it on, and a plausible null mood proxy carries no signal."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $V_t$ be month $t$'s aggregate musical valence and $r_t$ the market return.\n\n"
            "- **H₁ (lead-lag).** $r_t = \\alpha + \\beta\\, z(V_{t-1}) + \\varepsilon_t$ with "
            "$\\beta > 0$ at **HAC** $|t| \\ge 2$ (happier last month → higher this month).\n"
            "- **H₂ (robust lag).** The sign/significance survives a lag-1..5 sweep with a Bonferroni "
            "bar (no single cherry-picked lag).\n"
            "- **H₃ (not noise).** A circular-shift placebo does not reproduce the slope.\n"
            "- **H₄ (net).** A mood-timing rule beats buy-and-hold after costs.\n\n"
            "**Before any of these:** H₀-data — *does a real valence tape exist to test H₁?* It does "
            "**not** on a free stack, so H₁ is tested on a synthetic proxy and the Signal axis is "
            "**capped at NONE regardless of outcome**. We find **H₁ rejected** (wrong sign, "
            "insignificant), **H₂ rejected** (lone lag-2 is wrong-signed & below Bonferroni), "
            "**H₃ rejected** (placebo swallows it), **H₄ rejected** (rule trails the market)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **data wall**. Spotify's audio-features endpoint — the only public "
            "source of ``valence`` — was closed to new applications in **November 2024** and never "
            "exposed a survivorship-free historical monthly chart-valence panel. Edmans et al. (2022) "
            "used a **licensed proprietary** stream+valence dataset. So the honest retail verdict is "
            "not 'the claim is false' but 'the claim is **untestable** for free' — and the "
            "responsible move is to (a) prove the measurement engine is faithful on synthetic worlds, "
            "and (b) confirm a plausible *null* mood proxy shows no spurious edge. That is the same "
            "discipline as [256 Twitter-Mood](../../256-twitter-mood/), where the original GPOMS feed "
            "was also never public."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Regression.** OLS of $r_t$ on $z(V_{t-1})$ with a **Newey-West HAC** se (bandwidth "
            "$\\approx n^{1/4}$) — a persistent mood series autocorrelates, inflating the naive $t$.\n"
            "- **Lag sweep.** HAC $t$ for lags 1..5; compare the max $|t|$ to a Bonferroni bar "
            f"(~{R['bonf']}).\n"
            "- **Placebo.** 2000 circular shifts of $V$ against $r$ (preserving $V$'s "
            "autocorrelation); tail probability of the observed $|slope|$.\n"
            "- **Hit-rate.** $\\Pr(\\text{sign}(z(V_{t-1})) = \\text{sign}(r_t))$ vs the unconditional "
            "up-rate (the honest base rate).\n"
            "- **Trading.** Long when $V_{t-1}$ > trailing median (else cash / short), one-month "
            "execution lag, one-way costs × NAV on rebalances, short leg pays borrow; gross & net.\n"
            "- **Positive control.** Plant $\\beta > 0$ in the synthetic tape; average the HAC $t$ over "
            "25 seeds (house rule) and confirm ~0 at the null.\n\n"
            "Timing: the signal at month $t$ uses valence completed at the close of month $t-1$, so it "
            "is fully public before the return it predicts — the single documented execution lag. No "
            "future data enters the signal."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — H₁ & H₃\n\n"
            "Next-month return on lagged valence, HAC *t*, with the circular-shift placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.predictive_regression(FRAME, lag=1)\n"
            "    p = st.placebo_pvalue(FRAME, lag=1, n_perm=2000, seed=549)\n"
            "    slope, ols_t, hac_t, r2 = reg['slope'], reg['t'], reg['hac_t'], reg['r2']\n"
            "else:\n"
            f"    slope, ols_t, hac_t, r2, p = {R['slope']}, {R['ols_t']}, {R['hac_t']}, {R['r2']}, {R['placebo_p']}\n"
            "print(f'slope        {slope:+.4f}  (market return per +1 sigma valence)')\n"
            "print(f'plain OLS t  {ols_t:+.2f}')\n"
            "print(f'HAC t        {hac_t:+.2f}   <-- headline (needs >= +2 the RIGHT way for the claim)')\n"
            "print(f'R^2          {r2:.4f}')\n"
            "print(f'placebo p    {p:.3f}   (fraction of shuffles with |slope| >= observed)')\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['plain t','HAC t'], [ols_t, hac_t], color=[GREY, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1); ax.axhline(-2, ls='--', c=GREEN, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('t-stat on lagged-valence slope')\n"
            "ax.set_title(f'Wrong sign, inside noise (HAC t {hac_t:+.2f}, placebo p {p:.2f})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected** — the slope is **{R['slope']}** (a happier month "
            f"predicts a very slightly *weaker* market, the wrong sign) at HAC *t* **{R['hac_t']}**, and "
            f"H₃ **rejected** — the placebo *p* = **{R['placebo_p']}** puts the observed slope squarely "
            "inside the noise band. R² ≈ 0."
        ),
        md(
            "### 4b · The lag sweep — H₂ (the Granger multiple-comparisons trap)\n\n"
            "HAC *t* at each lead lag. Test enough lags and one will graze significance by chance."
        ),
        code(
            "sweep = " + repr(R["sweep"]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.lag_sweep(FRAME)\n"
            "else:\n"
            "    tab = pd.DataFrame([(s[0],s[1],s[2]) for s in sweep], columns=['lag','slope','hac_t']).set_index('lag')\n"
            "print(tab.round(4).to_string())\n"
            "mx = tab['hac_t'].abs().max()\n"
            f"print(f'\\nmax |HAC t| across 5 lags = {{mx:.2f}}  (Bonferroni bar ~ {R['bonf']}); "
            "sign of best lag =', 'NEGATIVE (wrong way)' if tab['hac_t'].abs().idxmax() and tab.loc[tab['hac_t'].abs().idxmax(),'hac_t']<0 else 'positive')\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "cols = [RED if abs(t)>=2 else GREY for t in tab['hac_t']]\n"
            "ax.bar([str(l) for l in tab.index], tab['hac_t'], color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1); ax.axhline(-2, ls='--', c=GREEN, lw=1)\n"
            f"ax.axhline({R['bonf']}, ls=':', c='k', lw=1); ax.axhline({-R['bonf']}, ls=':', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('lag (months valence leads)'); ax.set_ylabel('HAC t')\n"
            "ax.set_title('Lag 2 grazes -2 (dashed) but is below Bonferroni (dotted) and WRONG-signed')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected.** The lone sub-|2| exception is lag 2 (HAC *t* −2.11) "
            f"— *negative* (opposite the claim) and below the Bonferroni bar (~{R['bonf']}) for 5 "
            "comparisons. Cherry-pick the best lag and you data-mine a mirage."
        ),
        md(
            "### 4c · Hit-rate vs the honest base rate\n\n"
            "Directional accuracy of 'valence-up predicts market-up' against the unconditional up-rate."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hr = st.hit_rate(FRAME, lag=1); hit, base = hr['hit']*100, hr['base']*100\n"
            "else:\n"
            f"    hit, base = {R['hit']}, {R['base']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['mood-direction\\nhit-rate','always guess\\nthe majority'], [hit, base], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(50, ls='--', c=GREY, lw=1, label='coin flip'); ax.legend()\n"
            "ax.set_ylabel('% correct'); ax.set_ylim(0, 80)\n"
            "ax.set_title(f'Mood call {hit:.0f}% vs base rate {base:.0f}% (and below the 50% coin)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'hit-rate {hit:.1f}% vs base rate {base:.1f}% (gap {hit-base:+.1f} pts)')"
        ),
        md(
            f"> 💡 In plain words: the mood call is right only **{R['hit']}%** of the time — *below a "
            f"coin*, and far below the **{R['base']}%** you'd get by always predicting 'up' (markets "
            "rise most months). No directional edge."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — capped by data availability (no real valence tape); H₁/H₂/H₃ all "
            f"rejected on the honest proxy test (HAC *t* {R['hac_t']}, placebo *p* {R['placebo_p']}, "
            f"hit-rate {R['hit']}% < base {R['base']}%).\n"
            f"- **Tradability `MIRAGE`** — mood-timing +{R['lo_gross']}%/yr vs market "
            f"+{R['mkt_ann']}%/yr; long-short negative and worse after borrow.\n"
            "- **Music sentiment on the tape? `UNTESTABLE`** — real on a licensed dataset, unreachable "
            "free; the *engine* is verified instead."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the mood-timing rule, gross & net (H₄)\n\n"
            "Long when last month's valence is above its trailing median (else cash, or short for the "
            "long-short variant), one-month execution lag, costs × NAV, short leg pays borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lo = st.mood_timing(FRAME, long_short=False)\n"
            "    ls = st.mood_timing(FRAME, long_short=True)\n"
            "    lo_g, lo_n = lo['gross_ann']*100, lo['net_ann']*100\n"
            "    ls_g, ls_n = ls['gross_ann']*100, ls['net_ann']*100\n"
            "    mk = lo['mkt_ann']*100\n"
            "else:\n"
            f"    lo_g, lo_n, ls_g, ls_n, mk = {R['lo_gross']}, {R['lo_net']}, {R['ls_gross']}, {R['ls_net']}, {R['mkt_ann']}\n"
            "labels = ['long-only\\ngross','long-only\\nnet','long-short\\ngross','long-short\\nnet','BUY & HOLD\\nmarket']\n"
            "vals = [lo_g, lo_n, ls_g, ls_n, mk]\n"
            "cols = [AMBER, RED, RED, RED, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annual return %')\n"
            "ax.set_title('Every mood-timing variant trails simply holding the market')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-only  {lo_g:+.1f}% gross / {lo_n:+.1f}% net')\n"
            "print(f'long-short {ls_g:+.1f}% gross / {ls_n:+.1f}% net')\n"
            "print(f'buy & hold market {mk:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: H₄ **rejected.** Long-only makes **+{R['lo_gross']}%/yr** — barely a "
            f"third of just holding the market (**+{R['mkt_ann']}%/yr**) — and the long-short variant "
            f"is *negative* (**{R['ls_gross']}%** gross, **{R['ls_net']}%** net). Costs are a footnote; "
            "there's no edge to erode. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a genuine predictive "
            "edge (`predictive_beta > 0`) of increasing strength and watch the lag-1 HAC *t* climb "
            "past +2 — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "betas = [0.0, 0.005, 0.010, 0.015, 0.020]\n"
            "ts = [st.synthetic_mean_t(data, predictive_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted predictive_beta (0 = null)')\n"
            "ax.set_ylabel('mean lag-1 HAC t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted mood edge - flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'beta {b:+.3f} -> mean HAC t {t:+.2f}')"
        ),
        md(
            "The HAC *t* is ≈ 0 at the null and clears +2 by `predictive_beta` ≈ 0.007 — so the engine "
            "is a faithful detector with well-behaved size and power. The real-tape reading is "
            "therefore the *data* talking: a plausible, unprivileged mood series carries no market "
            "timing information, and the privileged version cannot be reached for free. For sibling "
            "mood proxies see [256 Twitter-Mood](../../256-twitter-mood/) and "
            "[300 Sports-Sentiment](../../300-sports-sentiment/)."
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
