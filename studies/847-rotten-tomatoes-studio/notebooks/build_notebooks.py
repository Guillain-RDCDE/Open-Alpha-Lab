"""Generate the two narrative notebooks for Study 847 (Rotten-Tomatoes -> Studio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY +
six-studio tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY + six studios
# 2021-06-01 -> 2026-06-30; 40 hardcoded major wide releases 2022-04-01 -> 2025-06-20).
R = dict(
    n_films=40, n_fresh=21, n_rotten=19, cal_lo="2022-04-01", cal_hi="2025-06-20",
    # opening-weekend [0..+1]: the clean, direct window
    ow_fresh_bps=-110.6, ow_fresh_t=-1.65, ow_rotten_bps=-98.8, ow_rotten_t=-1.45,
    ow_gap_bps=-11.9, ow_gap_t=-0.12,
    # following-week [+2..+6]
    fw_fresh_bps=181.1, fw_fresh_t=1.62, fw_fresh_nw=2.51,
    fw_rotten_bps=-634.6, fw_rotten_t=-2.14, fw_rotten_nw=-2.09,
    fw_gap_bps=815.7, fw_gap_t=2.58,
    # combined [0..+6]
    full_fresh_bps=70.5, full_fresh_t=0.67, full_rotten_bps=-733.3, full_rotten_t=-2.18,
    full_gap_bps=803.8, full_gap_t=2.28,
    # following-week hit rates
    hit_fresh_k=17, hit_fresh_n=21, hit_fresh_pct=81.0, hit_fresh_wilson=(60.0, 92.3),
    hit_rotten_k=14, hit_rotten_n=19, hit_rotten_pct=73.7, hit_rotten_wilson=(51.2, 88.2),
    # placebos
    perm_obs_bps=815.7, perm_mean_bps=0.6, perm_sd_bps=328.3, perm_p=0.001, perm_draws=20000,
    rand_obs_bps=-206.3, rand_mean_bps=-0.0, rand_sd_bps=119.2, rand_p=0.079, rand_draws=4000,
    # timer: cost -> (gross_bps, net_bps, t_net)
    timer={0.0: (396.5, 396.2, 2.56), 5.0: (396.5, 386.2, 2.50)},
    # window sweep: label -> (gap_bps, welch_t)
    windows={"pre [-6..-2]": (-210, -1.26), "pre [-11..-2]": (-26, -0.11),
             "open [0..+1]": (-12, -0.12), "week [+2..+6]": (816, 2.58),
             "[+7..+11]": (79, 0.33), "[+15..+20]": (-106, -0.70)},
    # robustness
    era_2223_gap=365, era_2223_t=2.03, era_2425_gap=1352, era_2425_t=2.18,
    loo_min_t=2.16, loo_max_t=2.84, trim_gap=480, trim_t=4.59,
    outlier_if=-4366, outlier_marley=-4050,
    # synthetic control
    syn_null_mean=-0.12, syn_null_sd=0.95, syn_null_fire=0,
    syn_plant_mean=4.49, syn_plant_sd=1.20, syn_plant_fire=20,
    syn_plant_seed_gap=349, syn_plant_seed_t=2.78,
    # fingerprints
    fp_panel="567671346662", fp_spy="d57404af997d", fp_dis="1f215beecb34",
    fp_wbd="ec1bfa8df210", fp_para="1c6e31d1e522", fp_cmcsa="88dadf4247a2",
    fp_nflx="a538858b0952", fp_sony="ba00ad7266b7",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Flop tanks studio%3F: Overstated](https://img.shields.io/badge/Flop%20tanks%20studio%3F-Overstated-8b949e?style=flat-square)\n\n"
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

from rotten_tomatoes import data, strategy as st

FILMS = data.film_table()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EVENTS = st.build_event_table(PRICES)
else:
    PRICES = EVENTS = None
print("real cache present:", HAVE_REAL, "| films in table:", len(FILMS),
      "|", int((FILMS['tier']=='fresh').sum()), "fresh /",
      int((FILMS['tier']=='rotten').sum()), "rotten")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a rotten movie tank the studio's stock? 🍅📉\n"
            "### The \"a flop sinks the studio / a hit lifts it\" story — tested on 40 big "
            "releases and 6 studios\n\n"
            + BADGES +
            "Every time a big movie gets panned, someone says the studio's stock should "
            "take a hit — and when a film is a critical darling, that it should pop. It's a "
            "tidy story: reviews are public, dated and loud, so surely they move the "
            "company that made the film.\n\n"
            "There's an obvious reason to doubt it: these \"studios\" are gigantic "
            "conglomerates. Disney is parks + ESPN + streaming; Comcast is your internet "
            "provider + Universal; Sony is PlayStation + camera sensors + music; one movie "
            "is a rounding error. So we tested it properly on **40 major releases "
            "(2022-2025)** across **six studios** (Disney, Warner Bros. Discovery, "
            "Paramount, Comcast/Universal, Netflix, Sony), each tagged **fresh** or "
            "**rotten** by critics.\n\n"
            "> 📓 **Want the *t*-stats, the placebos and the confound autopsy?** See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| On the **release weekend** — when the reviews are already out — do rotten "
            f"films' studios drop more than fresh ones'? | **No.** The fresh-minus-rotten "
            f"gap is **{R['ow_gap_bps']:+.1f} bps** (basically zero, and the *wrong* sign). "
            "The one window where a review effect should be sharpest shows nothing. |\n"
            "| In the **following week**, is there a gap? | **Yes — and it's real "
            f"in-sample** ({R['fw_gap_bps']:+.0f} bps, hard to get by luck). But it only "
            "shows up in *that one* week, it reverses afterward, and it's way too big to be "
            "a movie. |\n"
            "| So — does a flop tank the studio? | **Overstated.** What actually happens is "
            "subtler and not really about the film: struggling studios make *both* worse "
            "movies *and* worse stock returns in the same stretch. |\n"
            "| Could you trade it? | **Only on paper.** A fresh-long / rotten-short book "
            "\"works\" in this 40-trade backtest — but it's riding that one fragile window "
            "and one giant unrelated crash. |\n\n"
            "> A loud, intuitive story — and the clean version of the test comes back empty."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A movie's Rotten Tomatoes score is a public verdict on the studio's "
            "product. A rotten score is a flop in the making — sell the studio. A fresh "
            "score is a hit — buy it.\"*\n\n"
            "It sounds airtight because reviews really are public and dated. The catch is "
            "**scale**: a single film, even a $250M tentpole, is a tiny slice of a "
            "$100-200bn conglomerate's revenue — and a lot of the box-office outcome is "
            "already anticipated (tracking, pre-sales) before opening. So the efficient- "
            "markets prior is that a known, small, largely-expected event barely moves the "
            "parent."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were real *and* tradable, you'd have a clean event edge: read the "
            "Tomatometer on release day, go long the fresh studios and short the rotten "
            "ones. So we ask three things: does the studio move on the **release weekend**, "
            "does it move in the **following week**, and could you **trade** the split?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The films.** {R['n_films']} major wide releases {R['cal_lo']} → "
            f"{R['cal_hi']} ({R['n_fresh']} clearly-fresh ≥ 75, {R['n_rotten']} "
            "clearly-rotten < 50 — mixed 50-74 titles left out so the contrast is clean).\n"
            "- **The instrument.** The *distributing studio's* stock (DIS/WBD/PARA/CMCSA/"
            "NFLX/SONY), measured against the S&P 500 so we see the studio's *own* move, "
            "not the market's.\n"
            "- **Two windows.** The **opening weekend** (release day + next session) and "
            "the **following week** (the next five sessions).\n"
            "- **The luck check.** Shuffle the fresh/rotten labels 20,000 times — how often "
            "does a random relabelling produce a gap this big?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the clean test: the release weekend.** Reviews are already public, so "
            "if critics move the stock, it should show here. Fresh vs rotten:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ts = st.tier_stats(EVENTS, col='ow_car')\n"
            "    fb, rb = ts['fresh_bps'], ts['rotten_bps']\n"
            "else:\n"
            "    fb, rb = R['ow_fresh_bps'], R['ow_rotten_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.4))\n"
            "ax.bar(['fresh films\\n(studio)', 'rotten films\\n(studio)'], [fb, rb],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i, v in enumerate([fb, rb]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('studio opening-weekend abnormal return (bps)')\n"
            "ax.set_title('Release weekend: fresh and rotten studios move the SAME (≈ nothing extra)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'opening-weekend gap fresh-rotten = {fb-rb:+.1f} bps  (R: {R[\"ow_gap_bps\"]:+.1f})')"
        ),
        md(
            f"Both tiers wobble slightly *down* on release (a mechanical \"sell the "
            f"release\" tic every big film shares), and the **gap is "
            f"{R['ow_gap_bps']:+.1f} bps** — essentially zero, if anything the wrong sign. "
            "**On the window where a review effect should be loudest, there is none.**\n\n"
            "**Now the following week.** Here — and *only* here — a gap appears:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ts = st.tier_stats(EVENTS, col='fw_car')\n"
            "    fb, rb = ts['fresh_bps'], ts['rotten_bps']\n"
            "else:\n"
            "    fb, rb = R['fw_fresh_bps'], R['fw_rotten_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.4))\n"
            "ax.bar(['fresh films\\n(studio)', 'rotten films\\n(studio)'], [fb, rb],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i, v in enumerate([fb, rb]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('studio following-week abnormal return (bps)')\n"
            "ax.set_title('Following week: fresh drift up, rotten drift DOWN — a real in-sample gap')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'following-week gap fresh-rotten = {fb-rb:+.1f} bps  (R: {R[\"fw_gap_bps\"]:+.1f})')"
        ),
        md(
            f"That's a **{R['fw_gap_bps']:+.0f} bps** gap — rotten studios drift down ~6%, "
            "fresh drift up ~2% (market-adjusted) over the week. Shuffle the labels 20,000 "
            f"times and a gap this big shows up **{R['perm_p']*100:.1f}%** of the time. "
            "In-sample, that's a real association. **So why isn't it a win?**\n\n"
            "**Because the gap lives in exactly one window and nowhere else.** Watch the "
            "same fresh-minus-rotten gap slide across time relative to the release:"
        ),
        code(
            "labels = list(R['windows'].keys())\n"
            "gaps = [R['windows'][k][0] for k in labels]\n"
            "cols = [RED if abs(R['windows'][k][1]) >= 2 else GREY for k in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.5))\n"
            "ax.bar(labels, gaps, color=cols, width=.62)\n"
            "for i, k in enumerate(labels):\n"
            "    ax.annotate(f\"t={R['windows'][k][1]:+.2f}\", (i, gaps[i]), ha='center',\n"
            "                va='bottom' if gaps[i] >= 0 else 'top', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('fresh - rotten gap (bps)')\n"
            "ax.set_title('One window lights up (red); everything before and after is noise')\n"
            "plt.xticks(rotation=15); plt.tight_layout(); plt.show()\n"
            "print('the two windows BEFORE the release are flat -> not a slow build-up; '\n"
            "      'the gap has decayed to noise two weeks later -> it also reverses')"
        ),
        md(
            "The two windows **before** the release are flat, so it's not a slow build-up; "
            "and the gap is **gone** a week later and mildly reverses after that. A genuine "
            "\"the review moved the stock\" effect wouldn't hide on release weekend, appear "
            "for exactly five days, then vanish.\n\n"
            "**And the smoking gun:** the two biggest \"rotten drops\" aren't movies at "
            "all."
        ),
        code(
            "# the two most extreme rotten following-week 'drops' are Paramount M&A crashes\n"
            "if HAVE_REAL:\n"
            "    rot = EVENTS[(EVENTS['included']) & (EVENTS['tier']=='rotten')]\n"
            "    worst = rot.reindex(rot['fw_car'].astype(float).sort_values().index).head(4)\n"
            "    print('biggest rotten following-week moves (real tape):')\n"
            "    for r in worst.itertuples():\n"
            "        print(f'  {r.title:32s} {r.studio:5s} {r.date}  {r.fw_car*1e4:+.0f} bps')\n"
            "else:\n"
            "    print('IF (PARA, 2024-05-17):', R['outlier_if'], 'bps')\n"
            "    print('Bob Marley: One Love (PARA, 2024-02-14):', R['outlier_marley'], 'bps')\n"
            "print('\\nThose ~ -40% weeks are PARAMOUNT crashing on its 2024 Skydance-takeover'\n"
            "      ' drama — nothing to do with IF or Bob Marley being rotten.')"
        ),
        md(
            f"*IF* (−44%) and *Bob Marley: One Love* (−40%) \"rotten\" weeks are **Paramount "
            "crashing on 2024 takeover news** — pure confound. More generally, studios in "
            "structural decline (Warner Bros. Discovery, Paramount) tend to make *both* "
            "worse-reviewed films *and* weaker stocks in the same period. The tier isn't "
            "moving the price; it's a **flag for how the studio was already doing.**\n\n"
            "**Could you trade the split anyway?** On paper, yes — which is exactly why we "
            "grade tradability separately."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm0 = st.timer_stats(EVENTS, col='fw_car', cost_bps=0.0)\n"
            "    tm5 = st.timer_stats(EVENTS, col='fw_car', cost_bps=5.0)\n"
            "    g, n5, t5 = tm0['gross_bps'], tm5['net_bps'], tm5['t_net']\n"
            "else:\n"
            "    g, n5, t5 = R['timer'][0.0][0], R['timer'][5.0][1], R['timer'][5.0][2]\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.3))\n"
            "ax.bar(['gross', 'net (5 bps + borrow)'], [g, n5], color=[GREY, AMBER], width=.5)\n"
            "for i, v in enumerate([g, n5]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean return per event leg (bps)')\n"
            "ax.set_title(f'Long-fresh / short-rotten \"works\" in-sample — net t = {t5:+.2f} (40 trades)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net {n5:+.0f} bps/leg at 5 bps costs, t = {t5:+.2f} — but it rides the one '\n"
            "      'fragile window and the Paramount crash. Fragile, not investable.')"
        ),
        md(
            f"The book nets **{R['timer'][5.0][1]:+.0f} bps/leg** at *t* = "
            f"**{R['timer'][5.0][2]:.2f}** after costs — but it's a 40-trade backtest "
            "riding the one transient window and the one giant unrelated crash. That's a "
            "**Fragile** artifact, not a deployable edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** No gap on the release weekend (the clean window); a "
            "right-signed, significant gap only in the following week, but it's a single "
            "transient window, too big to be a movie, and driven by the studios' own "
            "fortunes (and one takeover crash). A real in-sample association — **not** a "
            "review effect.\n"
            "- **Tradability — Fragile.** The long-short book \"works\" on paper but rides "
            "the fragile window and the confound.\n"
            "- **\"Does a flop tank the studio?\" — Overstated.** The causal story isn't "
            "there; a smaller, murkier correlation is all that survives."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The honest limit is power and attribution.** 40 events, each a noisy "
            "five-day window, and a signal that can't be cleanly separated from \"which "
            "studios were sinking that year.\" A bigger sample and a studio-fixed-effects "
            "design (does a rotten film hurt *the same studio* vs its own average?) is the "
            "natural next step.\n"
            "- **Where a real version might live:** the *pure-play* film names (a studio "
            "that is mostly one franchise), or intraday reaction to the review-embargo lift "
            "rather than a five-day close-to-close window that swallows unrelated news.\n"
            "- **Sibling studies:** [771-box-office-bomb](../../771-box-office-bomb/) "
            "(sell Disney after a flop), [550-box-office-momentum](../../550-box-office-momentum/) "
            "(box-office revenue momentum), [296-oscars-effect](../../296-oscars-effect/) "
            "(the awards channel) and [552-app-store-rankings](../../552-app-store-rankings/) "
            "(product ratings for tech). Same \"public rating → stock\" question, different "
            "signals.\n\n"
            "*Think a studio-fixed-effects design on pure-play names would isolate a real "
            "review effect? Show it — out of sample, after costs — then we'll talk.*"
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
            "# Rotten-Tomatoes -> Studio — a quantitative teardown 🔬\n"
            "### Two pre-registered event windows · a Welch fresh-minus-rotten gap · "
            "tier-label vs random-date placebos (and why they disagree) · window-by-window "
            "transience · the Paramount-crash confound · a costed timer · a 20-seed "
            "synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — a film's Rotten-Tomatoes **tier** moves its **distributing "
            "studio's** stock around release — is a standard corporate event study. The job "
            "here is to measure it honestly on the modern tape, then explain why a "
            "*statistically significant, right-signed* gap is still stamped **Weak**.\n\n"
            "> ⚠️ **Data note.** SPY + DIS/WBD/PARA/CMCSA/NFLX/SONY total-return closes "
            "(2021→2026), yfinance, cached; **40 hardcoded major wide releases** 2022→2025 "
            "(clearly fresh ≥ 75 / rotten < 50 only). WBD trades from 2022-04-11, PARA is "
            "the post-2022-02 Paramount Global; **Netflix's event is the streaming "
            "premiere**, not an opening weekend — all named honestly. Methods in "
            "[`docs/references.md`](../docs/references.md); numbers in "
            "[`docs/results.md`](../docs/results.md) (panel fingerprint `" + R["fp_panel"] +
            "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | opening-weekend gap **{R['ow_gap_bps']:+.1f} bps** "
            f"(Welch *t* = {R['ow_gap_t']:+.2f}) — nothing; following-week gap "
            f"**{R['fw_gap_bps']:+.0f} bps** (Welch *t* = {R['fw_gap_t']:+.2f}, perm "
            f"*p* = {R['perm_p']:.3f}) but transient, non-causal |\n"
            f"| **Tradability** | `FRAGILE` | long-fresh/short-rotten net "
            f"**{R['timer'][5.0][1]:+.0f} bps/leg** (*t* = {R['timer'][5.0][2]:.2f}) — a "
            "40-trade artifact of the fragile window |\n"
            f"| **Flop tanks the studio?** | `OVERSTATED` | pooled-magnitude random-date "
            f"placebo *p* = **{R['rand_p']:.3f}** — a release week is no bigger than a "
            "random week; only the *split* is unusual |\n\n"
            "> 💡 In plain words: the clean, direct window is empty; the one window that "
            "isn't fails every check that would make it a believable review effect."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "For film $i$ with distributing studio $s(i)$ and first-tradable-session anchor "
            "$\\tau_i$, let $a^{s}_t = r^{s}_t - r^{SPY}_t - \\bar{c}^{s}$ be the studio's "
            "market-adjusted, mean-demeaned abnormal return (a $\\beta = 1$ market model "
            "with a constant-mean overlay). Define two window CARs: opening-weekend "
            "$\\mathrm{OW}_i = \\sum_{k=0}^{1} a^{s(i)}_{\\tau_i+k}$ and following-week "
            "$\\mathrm{FW}_i = \\sum_{k=2}^{6} a^{s(i)}_{\\tau_i+k}$.\n\n"
            "- **H₁ (direct).** $E[\\mathrm{OW} \\mid \\text{fresh}] > E[\\mathrm{OW} \\mid "
            "\\text{rotten}]$ — reviews are public pre-release, so the weekend should "
            "separate the tiers.\n"
            "- **H₂ (digest).** The same for $\\mathrm{FW}$ over the following week.\n"
            "- **H₃ (tradable).** A long-fresh/short-rotten book beats costs.\n\n"
            f"We find **H₁ rejected** (gap {R['ow_gap_bps']:+.1f} bps, Welch "
            f"*t* = {R['ow_gap_t']:+.2f}, wrong sign), **H₂ significant but not causal** "
            f"(gap {R['fw_gap_bps']:+.0f} bps, Welch *t* = {R['fw_gap_t']:+.2f} — real "
            "in-sample, fails the mechanism checks), **H₃ survives naively but Fragile.**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · Inference design — and why two placebos\n\n"
            "Releases are **independent, non-overlapping calendar dates**, so the primary is "
            "a **one-sample *t*** per tier and a **Welch *t*** on the fresh-minus-rotten "
            "gap. Two nulls, because they answer different questions:\n\n"
            "- **Tier-label permutation** (20 seeds × 1,000): keep every event's CAR, "
            "shuffle the fresh/rotten labels. Isolates whether the *split by tier* is "
            "unusual.\n"
            "- **Random-date** (20 seeds × 200): redraw pseudo-events on each studio's own "
            "tape. Isolates whether the *pooled magnitude* is unusual.\n\n"
            "> 💡 The tell: if a release genuinely *caused* the move, both should fire. Here "
            f"the split fires (*p* = {R['perm_p']:.3f}) but the pooled magnitude does **not** "
            f"(*p* = {R['rand_p']:.3f}) — the signature of a **confound correlated with "
            "tier**, not a release-driven move."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            f"- **Calendar.** {R['n_films']} releases {R['cal_lo']} → {R['cal_hi']} "
            f"({R['n_fresh']} fresh / {R['n_rotten']} rotten), hardcoded; tier bucket only.\n"
            "- **Tape.** SPY + six studios, total-return closes, 2021 → 2026-06-30 (as-of, "
            "last complete month).\n"
            "- **Windows.** Opening-weekend `[0..+1]` (direct) and following-week `[+2..+6]` "
            "(digest), both anchored at the first session on/after release.\n"
            "- **Headline.** Per-tier one-sample *t* + Welch gap *t* + Wilson hit rates.\n"
            "- **Nulls.** Tier-label permutation (the split) + random-date (the magnitude).\n"
            "- **Robustness.** Sub-era split, leave-one-studio-out, outlier trim, and a "
            "full window sweep from `[-11..-2]` to `[+15..+20]`.\n"
            "- **Timer.** Long-fresh/short-rotten on `[+2..+6]`; 2 × one-way cost × NAV per "
            "leg + borrow on shorts.\n"
            "- **Control.** Synthetic paired world, planted tier drift; the null must not "
            "fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two windows — direct vs digest\n\n"
            "Per-tier means with one-sample *t*'s, and the Welch gap, for both pre-"
            "registered windows."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for col, lab in [('ow_car','opening [0..+1]'), ('fw_car','following [+2..+6]'),\n"
            "                     ('full_car','combined [0..+6]')]:\n"
            "        ts = st.tier_stats(EVENTS, col=col)\n"
            "        rows.append((lab, ts['fresh_bps'], ts['fresh_t'], ts['rotten_bps'],\n"
            "                     ts['rotten_t'], ts['gap_bps'], ts['gap_welch_t']))\n"
            "else:\n"
            "    rows = [('opening [0..+1]', R['ow_fresh_bps'], R['ow_fresh_t'], R['ow_rotten_bps'],\n"
            "             R['ow_rotten_t'], R['ow_gap_bps'], R['ow_gap_t']),\n"
            "            ('following [+2..+6]', R['fw_fresh_bps'], R['fw_fresh_t'], R['fw_rotten_bps'],\n"
            "             R['fw_rotten_t'], R['fw_gap_bps'], R['fw_gap_t']),\n"
            "            ('combined [0..+6]', R['full_fresh_bps'], R['full_fresh_t'], R['full_rotten_bps'],\n"
            "             R['full_rotten_t'], R['full_gap_bps'], R['full_gap_t'])]\n"
            "df = pd.DataFrame(rows, columns=['window','fresh_bps','fresh_t','rotten_bps',\n"
            "                                 'rotten_t','gap_bps','gap_welch_t']).set_index('window')\n"
            "print(df.round(2).to_string())\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "x = np.arange(len(df)); w = 0.38\n"
            "ax.bar(x - w/2, df['fresh_bps'], width=w, color=GREEN, label='fresh')\n"
            "ax.bar(x + w/2, df['rotten_bps'], width=w, color=RED, label='rotten')\n"
            "for i, (f, r) in enumerate(zip(df['fresh_bps'], df['rotten_bps'])):\n"
            "    ax.annotate(f'{f:+.0f}', (i - w/2, f), ha='center', va='bottom' if f>=0 else 'top', fontsize=8)\n"
            "    ax.annotate(f'{r:+.0f}', (i + w/2, r), ha='center', va='bottom' if r>=0 else 'top', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(df.index)\n"
            "ax.set_ylabel('studio abnormal return (bps)')\n"
            "ax.set_title('The direct window (left) is flat; the split lives in the following week')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: opening-weekend gap **{R['ow_gap_bps']:+.1f} bps** (Welch "
            f"*t* = {R['ow_gap_t']:+.2f}) — H₁ rejected, wrong sign. Following-week gap "
            f"**{R['fw_gap_bps']:+.0f} bps** (Welch *t* = {R['fw_gap_t']:+.2f}); the "
            f"per-tier Newey-West *t*'s are {R['fw_fresh_nw']:+.2f} (fresh) and "
            f"{R['fw_rotten_nw']:+.2f} (rotten). That clears the bar — so we interrogate it."
        ),
        md(
            "### 4b · The two placebos — split vs magnitude\n\n"
            "The tier-label permutation (light in-notebook run; canonical 20k-draw *p* from "
            "`results.md`) against the random-date magnitude null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.permutation_placebo(EVENTS, col='fw_car', n_seeds=4, n_draws_per_seed=1000)\n"
            "    obs, draws = perm['obs_bps'], perm['draws_bps']\n"
            "else:\n"
            "    obs = R['perm_obs_bps']\n"
            "    rng = np.random.default_rng(847)\n"
            "    draws = rng.normal(R['perm_mean_bps'], R['perm_sd_bps'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: shuffled tier labels')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed gap {obs:+.0f} bps')\n"
            "ax.set_xlabel('fresh - rotten following-week gap under random relabelling (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"The SPLIT is unusual: canonical perm p = {R['perm_p']:.3f} (20k draws)\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"tier-label permutation: obs {R['perm_obs_bps']:+.0f} bps vs null \"\n"
            "      f\"{R['perm_mean_bps']:+.1f} (sd {R['perm_sd_bps']:.0f}) -> p = {R['perm_p']:.3f}\")\n"
            "print(f\"random-date (pooled MAGNITUDE): obs {R['rand_obs_bps']:+.0f} bps -> \"\n"
            "      f\"two-sided p = {R['rand_p']:.3f}  <-- a release week is NOT a big week\")"
        ),
        md(
            f"> 💡 In plain words: the **split** sits far in the tail (*p* = "
            f"{R['perm_p']:.3f}) — fresh really did out-return rotten here. But the **pooled "
            f"magnitude** is ordinary (*p* = {R['rand_p']:.3f}): release weeks aren't "
            "bigger-moving than random weeks. A causal release effect would make *both* "
            "unusual. Only the split being unusual is what a **tier-correlated confound** "
            "looks like."
        ),
        md(
            "### 4c · Window-by-window — the effect is one transient blip\n\n"
            "The fresh-minus-rotten gap swept from two weeks *before* the release to three "
            "weeks after. A genuine information effect would not be confined to a single "
            "post-release window."
        ),
        code(
            "labels = list(R['windows'].keys()); gaps = [R['windows'][k][0] for k in labels]\n"
            "tvals = [R['windows'][k][1] for k in labels]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in tvals]\n"
            "a1.bar(labels, gaps, color=cols, width=.62); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('fresh - rotten gap (bps)')\n"
            "a1.set_title('Flat before release, one red window at [+2..+6], noise after')\n"
            "a2.bar(labels, tvals, color=cols, width=.62); a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t'); plt.xticks(rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('windows (gap bps, t):', {k: R['windows'][k] for k in labels})"
        ),
        md(
            "> 💡 In plain words: the two **pre-release** windows are flat (no anticipation "
            "build-up), the gap is significant only at `[+2..+6]`, and it has decayed to "
            "noise by `[+7..+11]` and reverses by `[+15..+20]`. That is the shape of a "
            "look-elsewhere blip, not a durable reaction to news."
        ),
        md(
            "### 4d · The confound autopsy — Paramount, and \"bad studios make bad films\"\n\n"
            "The two most extreme rotten weeks are not films; and trimming them does not "
            "kill the split — it *strengthens* it, because the gap is a broad ambient tilt."
        ),
        code(
            "if HAVE_REAL:\n"
            "    inc = EVENTS[EVENTS['included']].copy(); inc['yr'] = inc['date'].str[:4].astype(int)\n"
            "    def gap(d):\n"
            "        f = d[d.tier=='fresh']['fw_car'].to_numpy(); r = d[d.tier=='rotten']['fw_car'].to_numpy()\n"
            "        return (f.mean()-r.mean())*1e4, st.welch_t(f, r)\n"
            "    g22, t22 = gap(inc[inc.yr<=2023]); g24, t24 = gap(inc[inc.yr>=2024])\n"
            "    trim = inc[inc.fw_car.abs() < 0.15]; gt, tt = gap(trim)\n"
            "    print(f'sub-era 2022-2023: gap {g22:+.0f} bps (t={t22:+.2f})')\n"
            "    print(f'sub-era 2024-2025: gap {g24:+.0f} bps (t={t24:+.2f})')\n"
            "    print(f'trim |CAR|>15% (drops {len(inc)-len(trim)}): gap {gt:+.0f} bps (t={tt:+.2f})')\n"
            "    print('leave-one-studio-out Welch t:')\n"
            "    for s in data.STUDIOS:\n"
            "        _, t = gap(inc[inc.studio != s]); print(f'   drop {s:5s}: t = {t:+.2f}')\n"
            "else:\n"
            "    print(f\"sub-era 2022-2023: gap {R['era_2223_gap']:+d} bps (t={R['era_2223_t']:+.2f})\")\n"
            "    print(f\"sub-era 2024-2025: gap {R['era_2425_gap']:+d} bps (t={R['era_2425_t']:+.2f})\")\n"
            "    print(f\"trim |CAR|>15%: gap {R['trim_gap']:+d} bps (t={R['trim_t']:+.2f})\")\n"
            "    print(f\"leave-one-studio-out Welch t in [{R['loo_min_t']:.2f}, {R['loo_max_t']:.2f}]\")\n"
            "print('\\nRobust to sub-era, LOO and trimming -> a real in-sample association.')\n"
            "print('But: IF (PARA) and Bob Marley (PARA) rotten weeks = 2024 Skydance-takeover crashes.')\n"
            "print('Declining studios (WBD, PARA) make BOTH worse films AND worse stocks -> selection.')"
        ),
        md(
            f"> 💡 In plain words: the split survives every robustness cut (sub-eras "
            f"*t* = {R['era_2223_t']:.2f} / {R['era_2425_t']:.2f}; leave-one-studio-out "
            f"*t* ∈ [{R['loo_min_t']:.2f}, {R['loo_max_t']:.2f}]; trimmed *t* = "
            f"{R['trim_t']:.2f}) — which is why it's **Weak, not None**. But it is a broad "
            "tilt tracking which studios were sinking, plus one literal takeover crash — "
            "which is why it's **Weak, not Real.** The tier is a proxy for the studio's "
            "fortunes, not a lever on the price."
        ),
        md(
            "### 4e · The timer — an honest cost check\n\n"
            "Long the fresh studio / short the rotten studio over the following week; 2 × "
            "one-way cost × NAV per leg plus 50 bps/yr borrow on the short (rotten) legs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for cb in (0.0, 5.0):\n"
            "        tm = st.timer_stats(EVENTS, col='fw_car', cost_bps=cb)\n"
            "        rows.append((cb, tm['gross_bps'], tm['net_bps'], tm['t_net']))\n"
            "else:\n"
            "    rows = [(cb, R['timer'][cb][0], R['timer'][cb][1], R['timer'][cb][2]) for cb in (0.0, 5.0)]\n"
            "for cb, g, n, t in rows:\n"
            "    print(f'cost {cb:>4.1f} bps/leg: gross {g:+.0f}  net {n:+.0f}  t(net) = {t:+.2f}')\n"
            "print('\\nSurvives naive costs (t = 2.50) — but it IS the fragile [+2..+6] split,')\n"
            "print('window-selected and inflated by the Paramount crash. Fragile, not investable.')"
        ),
        md(
            f"> 💡 In plain words: net **{R['timer'][5.0][1]:+.0f} bps/leg** at *t* = "
            f"**{R['timer'][5.0][2]:.2f}** — it \"passes\" only because it is the very same "
            "window-selected, non-causal gap, on 40 trades. A backtest artifact of a Weak "
            "signal is not a strategy."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic paired (studio, benchmark) world with a TUNABLE planted tier drift on "
            "the following-week window. The null (edge = 0) is checked over 20 seeds."
        ),
        code(
            "null_t = np.array([st.synthetic_detect(edge=0.0, seed=847 + s)['gap_welch_t'] for s in range(20)])\n"
            "plant_t = np.array([st.synthetic_detect(edge=0.004, seed=847 + s)['gap_welch_t'] for s in range(20)])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_t, color=GREY, s=40,\n"
            "           label='null (edge=0), 20 seeds')\n"
            "ax.scatter(np.ones(20) + np.linspace(-.12, .12, 20), plant_t, color=GREEN, s=40,\n"
            "           label='planted edge=0.004/day, 20 seeds')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x20', 'planted x20'])\n"
            "ax.set_ylabel('fresh - rotten gap Welch t')\n"
            "ax.set_title('Control: no null fires; a planted tier effect lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_t)>=2).sum()}/20')\n"
            "print(f'planted: mean t = {plant_t.mean():+.2f} (sd {plant_t.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(plant_t)>=2).sum()}/20')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"*t* = {R['syn_null_mean']:+.2f} and **never** fires; a planted tier drift "
            f"reads *t* = {R['syn_plant_mean']:+.2f} and fires {R['syn_plant_fire']}/20. The "
            "machinery is unbiased and faithful — and note its *modest* power at n≈20/tier, "
            "one more reason to treat a lone real-tape *t* = 2.58 with caution rather than "
            "as proof. *(A faithful-engine / power check only — never cited for the real-"
            "tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — opening-weekend gap **{R['ow_gap_bps']:+.1f} bps** "
            f"(Welch *t* = {R['ow_gap_t']:+.2f}, wrong sign) on the direct window; a "
            f"significant following-week gap **{R['fw_gap_bps']:+.0f} bps** (Welch "
            f"*t* = {R['fw_gap_t']:+.2f}, perm *p* = {R['perm_p']:.3f}, robust to sub-era & "
            "LOO) that fails the causal bar: no direct-window reaction, a single transient "
            f"window, pooled magnitude ordinary (*p* = {R['rand_p']:.3f}), implausible size, "
            "and a Paramount-takeover confound. A real in-sample association, **not** a "
            "review effect.\n"
            f"- **Tradability `FRAGILE`** — long-fresh/short-rotten net "
            f"**{R['timer'][5.0][1]:+.0f} bps/leg** (*t* = {R['timer'][5.0][2]:.2f}) is a "
            "40-trade artifact of the fragile window, not an edge.\n"
            "- **\"Does a flop tank the studio?\" `OVERSTATED`** — the causal story isn't "
            "supported; a smaller, harder-to-attribute correlation is all that survives at "
            f"n = {R['n_films']}."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Attribution is the real limit.** A studio-fixed-effects design (a rotten "
            "film vs *that studio's own* average, absorbing the ambient decline) and a "
            "larger sample would separate \"the review moved it\" from \"the studio was "
            "sinking anyway.\"\n"
            "- **Cleaner instruments:** pure-play film names, or an intraday window around "
            "the review-embargo lift instead of a five-day close-to-close CAR that swallows "
            "unrelated corporate news (see the Paramount crash).\n"
            "- **Dedup map:** [771-box-office-bomb](../../771-box-office-bomb/) (DIS flop "
            "event), [550-box-office-momentum](../../550-box-office-momentum/) (revenue "
            "momentum), [296-oscars-effect](../../296-oscars-effect/) (awards) and "
            "[552-app-store-rankings](../../552-app-store-rankings/) (tech product ratings) "
            "— same \"public rating → stock\" question, different signals.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
