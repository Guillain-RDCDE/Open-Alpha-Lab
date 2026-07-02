"""Generate the two narrative notebooks for Study 552 (App-Store-Rankings).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Everything is
deterministic and OFFLINE: this is a synthetic-only study (no free App Store ranking tape exists),
so every figure is drawn from the seeded synthetic panel in ../app_store_rankings/data.py. The dict
``R`` mirrors the frozen headline numbers in docs/results.md; ``HAVE_REAL`` is always False by
construction (fetch_panel cannot return real data), and the notebooks say so plainly.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; synthetic, seed 552,
# 48 months x 30 names). The honest headline is the NULL world (no real tape exists).
R = dict(
    fp_null="fc780c9f143d", fp_planted="5b92f4d9fdb5",
    n_months=48, n_names=30,
    # null world (the honest headline)
    null_ic=0.010, null_ic_t=0.39, null_hit=54.2,
    null_ls_gross=-0.53, null_ls_t=-0.06, null_net=-1.02, null_placebo=0.70,
    # planted-world ceiling (positive control at rank_alpha=0.15)
    plant_ic=0.080, plant_ic_t=3.00, plant_ls_gross=21.7, plant_net=21.2, plant_placebo=0.002,
    cost_bps=10.0, borrow_bps=100.0,
    # null robustness blocks: (label, mean_ic, t)
    win=[("months 0-11", 0.011, 0.20), ("months 12-23", -0.065, -1.13),
         ("months 24-35", 0.067, 1.46), ("months 36-47", 0.028, 0.56)],
    # seed-robust control: (rank_alpha, mean IC-t over 25 seeds)
    ctrl=[(0.0, 0.12), (0.05, 0.94), (0.10, 1.76), (0.15, 2.62), (0.20, 3.44), (0.30, 5.13)],
    # signal-noise sweep at rank_alpha=0.15: (signal_noise, mean IC-t)
    noise=[(0.5, 3.33), (1.0, 2.62), (2.0, 1.70), (4.0, 0.92)],
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

from app_store_rankings import data, strategy as st

# There is NO free App Store ranking tape (Apple has no historical rank API; vendor history is
# licensed/modelled). fetch_panel(fetch=False) returns empty -> HAVE_REAL is always False, and this
# study is synthetic-only. The honest headline is the seeded NULL world; the planted world is the
# positive control (never a real-tape banner).
HAVE_REAL = not data.fetch_panel(fetch=False).empty   # always False by construction
NULL = data.synthetic_history(rank_alpha=0.0, seed=552)       # honest headline (no signal)
PLANT = data.synthetic_history(rank_alpha=0.15, seed=552)     # positive control (signal planted)
print("real App-Store ranking tape present:", HAVE_REAL, "(synthetic-only study)")
print(f"null history fp {data.fingerprint(NULL)} | planted fp {data.fingerprint(PLANT)}"
      f" | {NULL['month'].nunique()} months x {NULL['ticker'].nunique()} names")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# App-Store Rankings — when the app climbs, does the stock climb? 📱\n"
            "### Turning 'this app is #1 today' into a stock signal, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Here's a tempting alt-data idea. When a public company's app **shoots up the App Store "
            "download charts**, that's a real-time hint that demand is booming — more downloads, more "
            "subscribers, more sales. If the stock market is slow to notice, then the companies whose "
            "apps *climbed the charts* should go on to **beat** the ones whose apps *sank*. It's the "
            "broad cousin of the desk's single-stock [Coinbase-Rank](../../294-coinbase-rank/) omen.\n\n"
            "So we test it the honest way: score a basket of consumer-tech names by how much their app "
            "*climbed the charts*, and see whether the climbers out-earn the sinkers.\n\n"
            "> ⚠️ **The catch, stated up front.** There is **no free, clean, historical App Store "
            "ranking database** you can download and line up against stock tickers — Apple only shows "
            "today's charts, and the usable history is expensive, licensed, and itself an *estimate*. "
            "So this study runs on a **synthetic** world we can control, to see what the signal *would* "
            "look like if it were real. That's why the best it can earn is **Weak**, never Real.\n"
            ">\n"
            "> 📓 Want the *t*-stats, the placebo test and the cost maths? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md). **Not investment advice.**"
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Could app-ranking momentum predict returns? | **Maybe — the idea is sound and our "
            "detector works.** When we *plant* a real effect, the engine catches it cleanly. |\n"
            "| Did we prove it on real stocks? | **No — and we can't.** There's no free, clean history "
            "of App Store ranks to test. So the honest run (a world with *no* signal) is flat, as it "
            "should be. |\n"
            "| Is there a hidden gotcha even if the data existed? | **Yes.** App rank is a *noisy* read "
            "on demand (it's capped at #1, and one app rarely equals one company's whole business). "
            "Make the read realistically noisy and the signal sinks below the bar. |\n"
            "| Could you trade it? | **No.** You'd have to short the *sinking-app* names — small, "
            "hard-to-borrow stocks — and rebalance every month. The book bleeds. |\n\n"
            "> Verdict: **Weak** signal (plausible, detectable, but unproven on any real tape and "
            "fragile to noise) × **Mirage** tradability."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When a public company's app climbs the App Store charts, the stock climbs with it.\"*\n\n"
            "The intuition: the App Store download rank is a **live demand meter**. If TikTok-parent, "
            "or a food-delivery app, or a fintech app is suddenly #3 instead of #40, something good is "
            "happening to its business *right now* — weeks or months before it shows up in a quarterly "
            "report. Buy the chart-climbers before Wall Street catches up. This is the alt-data "
            "'nowcasting' idea (Google-Trends-predicts-the-present, but for apps)."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be a genuine information edge — you'd be reading fundamentals "
            "off a public chart faster than the market. The desk has already looked at the "
            "single-name version ([294 Coinbase-Rank](../../294-coinbase-rank/): Coinbase tops the "
            "chart → crypto fades) and other soft alt-data feeds "
            "([257 AAII](../../257-aaii-sentiment/), [335 Buzz-ETF](../../335-buzz-sentiment-etf/)). "
            "Here we do the *broad cross-sectional* version: rank many consumer-tech names by how much "
            "their app climbed, and see who won."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score the climb.** For each company, measure how much its app *climbed the charts* "
            "over the last month (a positive number = it went up).\n"
            "2. **Sort.** Split the names into a *climbers* third and a *sinkers* third.\n"
            "3. **Compare next-month returns.** Did the climbers beat the sinkers? Do it every month "
            "and average.\n\n"
            "The honest snag: **we don't have real App Store history**, so we build a synthetic world "
            "where we can *turn the effect on or off* and check that our measuring stick works. Below, "
            "the honest headline is the world with the effect turned **off** (it should look like "
            "nothing) — and then we prove the engine *would* catch a real effect if one existed.\n\n"
            "*Timing:* the 'climb' is last month's change, known today; we buy today and hold one "
            "month. No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the honest run is *flat*\n\n"
            "With no real signal planted, do the chart-climbers beat the sinkers? They shouldn't — and "
            "they don't."
        ),
        code(
            "s = st.ic_summary(NULL); ls = st.long_short_summary(NULL)\n"
            "print(f\"honest (no-signal) world: mean IC {s['mean_ic']:+.3f} (t {s['t']:+.2f}), \"\n"
            "      f\"long-short {ls['gross']*100:+.2f}%/mo (t {ls['t']:+.2f})\")\n"
            "# show the monthly long-short spread wandering around zero\n"
            "lss = st.long_short_series(NULL)*100\n"
            "fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "ax.bar(lss.index, lss.values, color=[GREEN if v>0 else RED for v in lss.values], width=.8)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(lss.mean(), c=GREY, ls='--', lw=1.5, label=f'mean {lss.mean():+.2f}%')\n"
            "ax.set_xlabel('month'); ax.set_ylabel('climbers - sinkers, next-month %')\n"
            "ax.set_title('No real signal -> the climbers-minus-sinkers spread is just noise around 0')\n"
            "ax.legend(); fig.suptitle('SYNTHETIC null world (effect OFF)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> The mean is essentially zero (**{R['null_ls_gross']}%/mo**, *t* {R['null_ls_t']}). Good "
            "— that's exactly what a *no-signal* world should show. This is our baseline: any real edge "
            "would have to stand out from this."
        ),
        md(
            "**But does our measuring stick actually work?** Let's turn the effect *on* (a world where "
            "app-climb really does predict returns) and check the engine catches it — flat when off, "
            "rising when on, averaged over 25 different random worlds so no single lucky draw fools us."
        ),
        code(
            "alphas = " + repr([a for a, _ in R["ctrl"]]) + "\n"
            "ts = [st.synthetic_mean_ic_t(data, rank_alpha=a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('how strong we make the real effect'); ax.set_ylabel('detector score (IC t, 25 seeds)')\n"
            "ax.set_title('Flat when the effect is OFF, rising past the bar when we turn it ON')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'effect {a:.2f} -> detector score {t:+.2f}')"
        ),
        md(
            "So the engine is honest: **~0 when there's nothing there, past the bar when there is.** "
            "That means the flat headline is the *data* talking (no signal to find, because we don't "
            "have real ranks), not a broken tool."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The idea is plausible and our detector works, **but there's no free "
            "real App Store history to test it on**, so it can never earn 'Real'. And even a genuine "
            "effect sinks below the bar once the rank read is realistically noisy (next notebook).\n"
            "- **Tradability — Mirage.** You'd short small, hard-to-borrow sinking-app names and "
            "rebalance monthly; the book bleeds after costs."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even setting aside 'we can't get the data', the trade means **shorting the companies "
            "whose apps sank** — often the smallest, most fragile, hardest-to-borrow names — and doing "
            "it *every month*. In the honest world the spread is already negative before you pay a "
            "cent; add a monthly rebalance and a borrow fee and it only gets worse."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The single-name cousin.** [294 Coinbase-Rank](../../294-coinbase-rank/) tests the "
            "*contrarian top* version on one asset.\n"
            "- **The data wall.** The real test needs a paid, point-in-time, survivorship-clean rank "
            "panel joined to tickers — exactly what a retail stack can't reach.\n\n"
            "*Have licensed rank history? Fork this, feed a real panel into the same IC engine, and "
            "show the climbers-minus-sinkers spread clearing t = 2 after costs.*"
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
            "# App-Store Rankings — a quantitative teardown 🔬\n"
            "### Cross-sectional Spearman IC · IC *t*-stat · label-shuffle placebo · long-short (gross & net) · four-window sign wander · seed-robust positive control · signal-noise sweep\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We treat app-ranking "
            "improvement as a cross-sectional alpha signal and run the standard alt-data toolkit: the "
            "**information coefficient** (Spearman rank correlation of signal vs forward return) and "
            "its *t*-stat, a long-short tercile spread, a placebo null, and a seed-robust control.\n\n"
            "> ⚠️ **Synthetic-only, by data necessity.** No free, point-in-time, survivorship-clean "
            "App Store ranking panel exists (Apple has no historical rank API; vendor history is "
            "licensed and modelled). `fetch_panel(fetch=True)` *raises* rather than fabricate a tape. "
            "So `HAVE_REAL` is always False, the honest headline is the seeded **null world**, and the "
            "study is capped at `WEAK` (a `REAL` stamp needs a robust *t* >= 2 on a real tape). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | No real tape exists → no robust *t* >= 2 on real data. Honest "
            f"(null) world: mean IC **+{R['null_ic']}** (*t* **+{R['null_ic_t']}**), placebo *p* "
            f"{R['null_placebo']}. Engine is faithful (planted 0.15 → IC-*t* **+{R['plant_ic_t']:.2f}**; "
            f"null → **+{R['ctrl'][0][1]}**) but the effect sinks under realistic noise. |\n"
            f"| **Tradability** | `MIRAGE` | Monthly long-short, short leg = hard-to-borrow sinking-app "
            f"tail. Null world gross **{R['null_ls_gross']}%/mo** → net **{R['null_net']}%/mo** "
            f"(10 bps/leg × 4 + {R['borrow_bps']:.0f} bps/yr borrow). |\n\n"
            "> 💡 In plain words: the detector works, but the honest run is flat *because there is no "
            "real signal to feed it*, and even a planted one is fragile and un-tradable."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_i$ be firm $i$'s app **ranking-improvement** signal in a given month and $r_i$ its "
            "forward (next-month) return.\n\n"
            "- **H₁ (information coefficient).** The pooled cross-sectional Spearman "
            "$\\text{IC} = \\text{corr}(\\text{rank}(s), \\text{rank}(r)) > 0$ at *t* >= 2.\n"
            "- **H₂ (long-short).** Long the top-tercile climbers, short the bottom-tercile sinkers "
            "earns a positive spread at *t* >= 2.\n"
            "- **H₃ (robustness).** The IC keeps its sign across time blocks.\n"
            "- **H₄ (net).** H₂ survives a monthly rebalance and the sinking-app short borrow.\n\n"
            "We **cannot test H₁–H₄ on a real tape** (none exists). On the honest **null** world all "
            "four are flat (correct); the **synthetic control** shows the engine *would* accept H₁/H₂ "
            "if a real effect were present — but H₄ fails even then, and H₁ fails under realistic noise."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. App-download rank is a legitimate demand nowcast in the "
            "alt-data literature (Da-Engelberg-Gao attention; Froot-et-al real-time sales). Two forces "
            "cap it here: (a) **no free tape** — the usable rank history is licensed/estimated, so a "
            "retail replication is impossible; and (b) **a noisy read** — rank is discrete, capped at "
            "#1, and one app rarely equals one public company's whole revenue, so the signal-to-noise "
            "is low. That maps to the desk's other soft-alt-data studies "
            "([335 Buzz-ETF](../../335-buzz-sentiment-etf/), [392 Glassdoor](../../392-glassdoor-sentiment/))."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $s$ = z-scored ranking improvement across the cross-section each month "
            "(a *trailing* change, public at the observation close — no look-ahead).\n"
            "- **IC.** Cross-sectional Spearman rank corr of $s$ vs forward return, one per month; "
            "the pooled mean IC and its *t* (mean / SE over months) are the headline.\n"
            "- **Long-short.** Top-tercile minus bottom-tercile forward return, monthly.\n"
            "- **Placebo.** Within-month label shuffle (2000 perms) → the mean IC's tail probability.\n"
            "- **Robustness.** Four disjoint month blocks; read the IC sign.\n"
            "- **Frictions.** 4 one-way cost crossings/month (both legs, entry+exit) + a monthly "
            "borrow on the short leg.\n"
            "- **Positive control.** A deterministic synthetic world with a planted effect "
            "(`rank_alpha > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the signal is a trailing ranking change (public at the observation close); we "
            "enter at that close and hold the forward month. One documented execution lag; on a monthly "
            "horizon a one-bar-later entry moves the headline IC negligibly and changes no stamp."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The information coefficient — H₁\n\n"
            "The monthly IC distribution and its pooled *t*, on the honest (null) world."
        ),
        code(
            "ics = st.monthly_ic(NULL)\n"
            "s = st.ic_summary(NULL); p = st.placebo_pvalue(NULL, n_perm=2000, seed=552)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.hist(ics.values, bins=14, color=GREY, edgecolor='white')\n"
            "ax.axvline(0, c='k', lw=1); ax.axvline(s['mean_ic'], c=RED, lw=2, label=f\"mean IC {s['mean_ic']:+.3f}\")\n"
            "ax.set_xlabel('monthly Spearman IC'); ax.set_ylabel('months')\n"
            "ax.set_title(f\"Honest world: mean IC {s['mean_ic']:+.3f} (t {s['t']:+.2f}), placebo p {p:.3f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"mean IC {s['mean_ic']:+.3f} | t {s['t']:+.2f} | hit-rate {s['hit']*100:.1f}% | placebo p {p:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: H₁ **flat** on the honest world — mean IC **+{R['null_ic']}** "
            f"(*t* **+{R['null_ic_t']}**), placebo *p* {R['null_placebo']}. No information, as it "
            "should be when no signal is planted. (There is no real tape to do better on.)"
        ),
        md(
            "### 4b · The long-short — H₂ & H₄ (gross and net)\n\n"
            "Top-tercile climbers minus bottom-tercile sinkers, and what costs + borrow do to it."
        ),
        code(
            "ls = st.long_short_summary(NULL, cost_bps=10.0, borrow_ann_bps=100.0)\n"
            "gross, net, t = ls['gross']*100, ls['net']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('climbers - sinkers, %/mo')\n"
            "ax.set_title(f'Negative before costs ({gross:+.2f}%/mo, t {t:+.2f}), worse after ({net:+.2f}%/mo)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f}%/mo (t {t:+.2f}) -> net {net:+.2f}%/mo (10 bps/leg x4 + 100 bps/yr borrow)')"
        ),
        md(
            f"> 💡 In plain words: on the honest world the spread is **{R['null_ls_gross']}%/mo** before "
            f"costs and **{R['null_net']}%/mo** after — nothing to harvest. The short leg (sinking-app "
            "names) is exactly the expensive-to-borrow tail. `MIRAGE`."
        ),
        md(
            "### 4c · Robustness — H₃ (does the IC hold its sign?)\n\n"
            "The mean IC over four disjoint month blocks (honest world)."
        ),
        code(
            "rows = st.robustness_windows(NULL, 4)\n"
            "tab = pd.DataFrame(rows).set_index('label')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "cols = [GREEN if v>0 else RED for v in tab['mean_ic']]\n"
            "ax.bar(range(len(tab)), tab['mean_ic'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index, rotation=15)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean IC')\n"
            "ax.set_title('The IC wanders around zero, flipping sign block to block (noise)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ is moot on the null world — the IC just wanders around zero and "
            "flips sign, which is what noise looks like. (On a real tape this is the check a bankable "
            "signal must pass; we have no real tape to run it on.)"
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — no real tape → no robust *t* >= 2 on real data (capped by rule). "
            f"Honest world flat (IC *t* +{R['null_ic_t']}, placebo *p* {R['null_placebo']}); engine "
            "faithful but effect fragile to noise.\n"
            f"- **Tradability `MIRAGE`** — monthly rebalance, sinking-app short borrow; null gross "
            f"{R['null_ls_gross']}%/mo → net {R['null_net']}%/mo."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real effect "
            "(`rank_alpha > 0`) of increasing strength and watch the seed-averaged IC-*t* rise past "
            "the bar — 25 seeds so no single lucky draw can fake it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "alphas = [a for a,_ in ctrl]\n"
            "ts = [st.synthetic_mean_ic_t(data, rank_alpha=a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted rank_alpha (return loading on the true demand signal)')\n"
            "ax.set_ylabel('mean IC-t (25 seeds)')\n"
            "ax.set_title('Flat at the null, past the bar as the planted effect grows — a faithful detector')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'rank_alpha {a:.2f} -> mean IC-t {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: IC-*t* is **+{R['ctrl'][0][1]}** at the null and climbs past 2 by "
            f"`rank_alpha` ≈ 0.15 (**+{R['plant_ic_t']:.2f}**). The engine banks a real effect — so the "
            "flat headline is the absence of a real signal, not a broken tool."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the noise that sinks a *real* effect\n\n"
            "Even if the data existed, app rank is a **noisy** read on demand (discrete, capped at #1, "
            "one app ≠ one company). Hold the true effect fixed at `rank_alpha` = 0.15 and crank the "
            "read-noise up: the same effect falls **below** the bar."
        ),
        code(
            "noise = " + repr(R["noise"]) + "\n"
            "sns = [s for s,_ in noise]\n"
            "ts = [st.synthetic_mean_ic_t(data, rank_alpha=0.15, n_seeds=25, signal_noise=s) for s in sns]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(sns, ts, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('signal_noise (how blurry the rank read is)')\n"
            "ax.set_ylabel('mean IC-t (25 seeds), true effect fixed at 0.15')\n"
            "ax.set_title('A genuine effect sinks below the bar as the rank read gets noisier')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for s, t in zip(sns, ts): print(f'signal_noise {s:.1f} -> mean IC-t {t:+.2f}')"
        ),
        md(
            "The same *true* effect clears the bar with a clean read (`signal_noise` 0.5 → *t* "
            f"+{R['noise'][0][1]}) but **fails** with a noisy one (`signal_noise` 4.0 → *t* "
            f"+{R['noise'][-1][1]}). App rank is closer to the noisy end. So even the optimistic case "
            "is fragile. For the single-name cousin see "
            "[294 Coinbase-Rank](../../294-coinbase-rank/); for other soft-alt-data feeds, "
            "[335 Buzz-ETF](../../335-buzz-sentiment-etf/) and [392 Glassdoor](../../392-glassdoor-sentiment/)."
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
