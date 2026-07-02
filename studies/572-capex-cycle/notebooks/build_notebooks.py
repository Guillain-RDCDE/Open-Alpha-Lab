"""Generate the two narrative notebooks for Study 572 (Capex-Cycle).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached prices + cash-flow
snapshot under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; scored FY2024,
# forward 2025-06-27 -> 2026-06-26; 42 survivors; panel fp 5a4b0aebc102).
R = dict(
    fp_px="a01c8032c6fc", fp_cf="8a66058175f0", fp_panel="5a4b0aebc102",
    n=42, k=13, split="2025-06-27", end="2026-06-26", upto=2024,
    harvest_mean=67.0, binge_mean=77.1, spread=-10.2, hedge_t=-0.14,
    ic=0.004, ic_t=0.03, spearman=-0.21, placebo_p=0.88,
    net=-11.4, cost_bps=5.0, borrow_bps=100.0,
    # robustness windows: (label, IC, IC_t, spread%)
    win=[("FY24 -> 2025-06->2026-06", 0.004, 0.03, -10.2),
         ("FY24 -> 2025-06->2025-12", 0.175, 1.12, -2.7),
         ("FY24 -> 2025-01->2026-01", 0.146, 0.94, -20.4),
         ("FY25 -> 2026-02->2026-06", -0.018, -0.12, -0.2)],
    # synthetic control: (binge_alpha, mean IC-t over 25 seeds)
    ctrl=[(0.0, -0.15), (-0.10, -0.94), (-0.20, -1.72), (-0.30, -2.49), (-0.50, -4.01)],
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

from capex_cycle import data, strategy as st

SPLIT, END, UPTO = "2025-06-27", "2026-06-26", 2024

def load_real():
    \"\"\"Cache-first real cross-section (empty frame offline).\"\"\"
    px = data.fetch_prices(fetch=False)
    cf = data.fetch_cashflow(fetch=False)
    if px.empty or not cf:
        return pd.DataFrame()
    return st.build_panel(px, cf, split_date=SPLIT, end_date=END, upto_year=UPTO)

PANEL = load_real()
HAVE_REAL = not PANEL.empty
print("real capex-cycle panel present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(PANEL)} names, scored FY{UPTO} -> {SPLIT}..{END})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Capex Cycle — do spending *bingers* underperform the *harvesters*? 🏗️\n"
            "### Sorting big companies by whether they're ramping or cutting capital spending, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Capex_cousin_of_asset_growth%3F: Busted](https://img.shields.io/badge/Capex_cousin_of_asset_growth%3F-Busted-8b949e?style=flat-square)\n\n"
            "There's a well-worn idea in finance: companies that go on a **spending binge** — pouring "
            "money into new factories, data-centres, mines — tend to *disappoint* their shareholders "
            "afterwards. Managers over-build, the market gets over-excited, and the stock lags. The "
            "flip side: companies *harvesting* cash (letting spending taper off) quietly do better. "
            "It's the investment cousin of the famous **asset-growth** anomaly.\n\n"
            "So we test it the cheap way: take a basket of big-name stocks, measure whether each one "
            "is **ramping** or **cutting** its capital spending (relative to its size), and see "
            "whether the harvesters beat the bingers over the next year.\n\n"
            "> 📓 **This is the plain-language layer.** Want the correlation stats, the placebo test "
            "and the cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the *bingers* earn less (the anomaly)? | **No.** Over 2025-26 the binge bucket "
            f"earned **+{R['binge_mean']}%** vs the harvest bucket's **+{R['harvest_mean']}%** — the "
            "bingers came out *ahead*. |\n"
            f"| Is there any signal at all? | **No.** The correlation between spending-ramp and future "
            f"return is **+{R['ic']}** — a dead zero (*t* = +{R['ic_t']}). |\n"
            f"| Is it just luck? | The shuffle test says *p* = {R['placebo_p']} — the small gap is "
            "pure noise. |\n"
            "| Could you trade it? | **No.** No edge, faint wrong sign, and the bingers you'd short "
            "are the crowded AI-datacentre names. |\n\n"
            "> The anomaly is real in the textbooks — but the textbook version needs the over-builders "
            "that *failed*. Our basket only has survivors, and 2024-25 was an AI-capex boom that paid "
            "the biggest spenders (Amazon, Google, Microsoft, Meta)."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Companies ramping their capital spending earn lower future returns than companies "
            "harvesting cash.\"* — the capex/investment anomaly (Titman-Wei-Xie 2004; the cousin of "
            "Cooper-Gulen-Schill asset growth).\n\n"
            "The intuition: when a company floods money into new capacity, managers tend to "
            "over-invest (empire-building), and investors over-extrapolate the growth story. The new "
            "factories don't earn what everyone hoped, and the stock drifts down. The *disciplined* "
            "firm that returns cash instead quietly outperforms. We look at the **change** in "
            "spending intensity — the phase of the capex *cycle* — a binge if it's rising, a harvest "
            "if it's falling."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it's a simple rule: *avoid (or short) the empire-builders, own the "
            "cash-harvesters*. The desk has already tested the two neighbours — total-asset growth "
            "([244 Asset-Growth](../../244-asset-growth/)) and the capex *level* "
            "([523 Investment-To-Assets](../../523-investment-to-assets/)). Here we go at the capex "
            "**growth** — the ramp itself — to see if the *change* in spending carries a signal the "
            "level and the total don't."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the ramp.** For each firm: capex ÷ assets this year, minus capex ÷ assets "
            "last year. Rising = a **binge**; falling = a **harvest**.\n"
            "2. **Sort.** Split the basket into a *harvest* third (cutting) and a *binge* third "
            "(ramping).\n"
            "3. **Compare forward returns.** Did the harvest third beat the binge third over the next "
            "year? The anomaly says yes.\n\n"
            "Catch: our basket is companies **still large in 2026** — so the over-builders that "
            "actually blew up (the heart of the real effect) are missing. And yfinance only gives us "
            "~4 years of statements, so this is a *snapshot*, not a deep history.\n\n"
            "*Timing:* the ramp uses only fiscal-2024 statements (public by mid-2025); we enter at "
            "the 2025-06 close and hold a year. No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Who won — the harvest third or the binge third?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.bucket_returns(PANEL, frac=0.3); h = st.hedge_tstat(b)\n"
            "    harv, bing, spr, t = b['harvest_mean']*100, b['binge_mean']*100, b['spread']*100, h['t']\n"
            "    banner = f'REAL tape ({len(PANEL)} survivors, FY2024 signal, 2025-06 -> 2026-06)'\n"
            "else:\n"
            f"    harv, bing, spr, t = {R['harvest_mean']}, {R['binge_mean']}, {R['spread']}, {R['hedge_t']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['HARVEST third','BINGE third'], [harv, bing], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'The anomaly says HARVEST wins. Here the BINGE third won (+{bing:.0f}% vs +{harv:.0f}%)')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Harvest +{harv:.1f}%  vs  Binge +{bing:.1f}%  ->  spread {spr:+.1f}% (t {t:+.2f})')"
        ),
        md(
            f"There it is: **no edge, faint wrong way.** The anomaly predicts the *harvest* names win. "
            f"On 2025-26 the **binge** names edged ahead (+{R['binge_mean']}% vs +{R['harvest_mean']}%) "
            f"— the harvest-minus-binge spread is **{R['spread']}%** with a *t* of just "
            f"**{R['hedge_t']}** (statistically nothing). Why? The biggest bingers here are the AI "
            "hyperscalers building data-centres — and they led the market."
        ),
        md(
            "**Is there a signal hiding across different windows?** Run the same sort over several "
            "forward windows and read the correlation (IC) between spending-ramp and return:"
        ),
        code(
            "wins = " + repr([(w[0], w[3]) for w in R["win"]]) + "\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_prices(fetch=False); cf = data.fetch_cashflow(fetch=False)\n"
            "    spans = [('2025-06-27','2026-06-26',2024),('2025-06-27','2025-12-27',2024),\n"
            "             ('2025-01-02','2026-01-02',2024),('2026-02-02','2026-06-26',2025)]\n"
            "    labs = ['FY24\\n25-06>26-06','FY24\\n25-06>25-12','FY24\\n25-01>26-01','FY25\\n26-02>26-06']\n"
            "    spr = []\n"
            "    for (sp,en,uy) in spans:\n"
            "        pn = st.build_panel(px, cf, split_date=sp, end_date=en, upto_year=uy)\n"
            "        spr.append(st.bucket_returns(pn, 0.3)['spread']*100 if not pn.empty else np.nan)\n"
            "else:\n"
            "    labs = ['FY24\\n25-06>26-06','FY24\\n25-06>25-12','FY24\\n25-01>26-01','FY25\\n26-02>26-06']\n"
            "    spr = [w[1] for w in wins]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in spr]\n"
            "ax.bar(range(len(labs)), spr, color=cols, width=.55)\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('harvest - binge spread %')\n"
            "ax.set_title('No window shows a clean harvest-wins (green) edge — all small, mostly red')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by window:', [round(s,1) for s in spr])"
        ),
        md(
            "So there's nothing to grab. Every window is a small spread that leans the *wrong* way "
            "(binge wins) or is a coin-flip. A 'signal' this small and unstable isn't something you "
            "can bet on."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The correlation between spending-ramp and return is +{R['ic']} "
            f"(*t* +{R['ic_t']}); the hedge is the wrong sign ({R['spread']}%, *t* {R['hedge_t']}) and "
            "unstable. No bankable capex-cycle edge here.\n"
            "- **Tradability — Mirage.** A survivor snapshot, and the binge leg you'd short is the "
            "crowded AI-capex tail. Nothing to harvest.\n"
            "- **Capex cousin of asset growth? — Busted.** Like its neighbours (244, 523), the "
            "capex-growth channel vanishes on a survivor large-cap basket."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even ignoring the wrong sign, the trade would mean *shorting* the biggest spenders — "
            "exactly today's crowded, expensive-to-borrow AI-datacentre names — on a basket that has "
            "already quietly deleted every company that over-built and failed. The real anomaly lives "
            "in the firms that *blew up*; a blue-chip survivor snapshot can't see them."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The neighbours.** [244 Asset-Growth](../../244-asset-growth/) is the *total*-asset "
            "version; [523 Investment-To-Assets](../../523-investment-to-assets/) is the capex "
            "*level*. This study is the capex *growth* (the ramp).\n"
            "- **The real test needs the dead and a deep panel.** Re-run this with delisted/failed "
            "over-builders included and a Compustat-depth history over a full cycle — that's where "
            "the anomaly earns its keep.\n\n"
            "*Think the capex-cycle effect is real and we just missed it? Fork this, add failed "
            "capex-heavy firms and a 2000-2002 window, and show the harvest-minus-binge spread "
            "clearing t = 2 the right way.*"
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
            "# The Capex Cycle — a quantitative teardown 🔬\n"
            "### capex-cycle signal · cross-sectional IC (with t) · tercile hedge two-sample *t* · label-shuffle placebo · four-window sweep · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Capex_cousin_of_asset_growth%3F: Busted](https://img.shields.io/badge/Capex_cousin_of_asset_growth%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build a capex-*cycle* "
            "signal (the change in capex intensity), and test whether the bingers earn less (the "
            "anomaly) — or, here, nothing.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily prices + a cash-flow snapshot, "
            f"42 survivors (as-of 2026-06-30, panel fp `{R['fp_panel']}`); the offline core and tests "
            "run on a deterministic synthetic world. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | IC **+{R['ic']}** (*t* **+{R['ic_t']}**); hedge spread "
            f"**{R['spread']}%** (two-sample *t* **{R['hedge_t']}**, placebo *p* {R['placebo_p']}) — "
            "no edge, faint wrong sign, sign unstable; yfinance **snapshot** caps below REAL. |\n"
            f"| **Tradability** | `MIRAGE` | Survivor snapshot, binge short = crowded AI-capex borrow. "
            f"Gross {R['spread']}% → net {R['net']}% (5 bps/leg + {R['borrow_bps']:.0f} bps borrow). |\n"
            f"| **Capex cousin of asset growth?** | `BUSTED` | Vanishes like [244](../../244-asset-growth/) "
            "and [523](../../523-investment-to-assets/) on the survivor large-cap panel. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on this "
            "survivor snapshot the capex-cycle effect is a flat zero, faint wrong sign, sign-unstable."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $C_i$ be firm $i$'s capex_cycle (the change in capex intensity) and $r_i$ its "
            "forward return.\n\n"
            "- **H₁ (the IC).** $\\text{corr}(C, r) < 0$ at *t* ≥ 2 (bingers underperform).\n"
            "- **H₂ (the hedge).** $\\mathbb{E}[r \\mid \\text{harvest}] - "
            "\\mathbb{E}[r \\mid \\text{binge}] > 0$ (long-harvest / short-binge pays).\n"
            "- **H₃ (robustness).** The sign of H₁/H₂ is stable across windows.\n"
            "- **H₄ (net).** H₂ survives costs and the binge-leg borrow.\n\n"
            "We find **H₁ rejected** (IC ≈ 0, *t* +0.03), **H₂ rejected** (spread wrong sign, "
            "*t* −0.14), **H₃ rejected** (small, unstable), **H₄ moot** (no gross edge)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The capex/investment anomaly is documented on broad, "
            "survivorship-free data driven by firms that *over-build and fail*. Two forces kill it "
            "here: (a) **survivorship** — the basket already dropped the failed over-builders, "
            "removing the left tail; and (b) a **hyperscaler-capex melt-up** (2024-25) that rewarded "
            "the biggest ramps (AMZN, GOOGL, MSFT, META building AI datacentres). That maps to the "
            "same flat readings on [244 Asset-Growth](../../244-asset-growth/) and "
            "[523 Investment-To-Assets](../../523-investment-to-assets/) on the same survivor panel."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $C = \\frac{|CapEx_t|}{Assets_{t-1}} - \\frac{|CapEx_{t-1}|}{Assets_{t-2}}$ "
            "(a *binge* if positive), scored as-of FY2024.\n"
            "- **Headline IC.** Pearson corr$(C, r)$ with $t = r\\sqrt{(n-2)/(1-r^2)}$; plus the "
            "rank (Spearman) IC.\n"
            "- **Hedge.** Tercile tails (≈13 names each): harvest (lowest $C$) − binge (highest $C$), "
            "two-sample (Welch) *t*.\n"
            "- **Placebo.** Label-shuffle null on the hedge spread (2000 perms).\n"
            "- **Robustness.** Re-score over four forward/fiscal windows.\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the binge short.\n"
            "- **Positive control.** A deterministic synthetic world with a planted effect "
            "(`binge_alpha < 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: capex_cycle is scored from the last two reported fiscal years (public at the "
            "split close); we enter at that close and hold the forward window. No future data enters "
            "the signal (no look-ahead). The same-close fill is one documented lag."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline IC — H₁\n\n"
            "The cross-sectional correlation between capex_cycle and forward return, with a *t*-stat."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ic = st.information_coefficient(PANEL)\n"
            "    x = PANEL['capex_cycle']; y = PANEL['forward_ret']*100\n"
            "    r, r_t, sp = ic['ic'], ic['ic_t'], ic['spearman']\n"
            "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "    ax.scatter(x, y, s=30, color=GREY)\n"
            "    m, bb = np.polyfit(x, y, 1)\n"
            "    xs = np.linspace(x.min(), x.max(), 50)\n"
            "    ax.plot(xs, m*xs + bb, c=RED, lw=2)\n"
            "    ax.set_xlabel('capex_cycle (higher = bigger binge)'); ax.set_ylabel('forward return %')\n"
            "    ax.set_title(f'IC (Pearson) = {r:+.3f}  (t {r_t:+.2f}) — a NEGATIVE IC would be the anomaly')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    r, r_t, sp = {R['ic']}, {R['ic_t']}, {R['spearman']}\n"
            "print(f'Pearson IC {r:+.3f} (t {r_t:+.2f}) | Spearman IC {sp:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected.** The Pearson IC is **+{R['ic']}** (*t* +{R['ic_t']}) "
            f"— a dead zero. The rank IC is mildly negative (Spearman {R['spearman']}), the only whiff "
            "of the predicted direction, but it's not significant on 42 names and is just a handful of "
            "mega-cap bingers that were also mega-winners (Pearson washes it out)."
        ),
        md(
            "### 4b · The tercile hedge — H₂\n\n"
            "Long the harvest tercile, short the binge tercile; two-sample *t* and placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.bucket_returns(PANEL, 0.3); h = st.hedge_tstat(b)\n"
            "    p = st.placebo_pvalue(PANEL, n_perm=2000, seed=572)\n"
            "    harv, bing, spr, t = b['harvest_mean']*100, b['binge_mean']*100, b['spread']*100, h['t']\n"
            "else:\n"
            f"    harv, bing, spr, t, p = {R['harvest_mean']}, {R['binge_mean']}, {R['spread']}, {R['hedge_t']}, {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['HARVEST','BINGE'], [harv, bing], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Harvest - binge = {spr:+.1f}%  (two-sample t {t:+.2f}, placebo p {p:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Harvest +{harv:.1f}% | Binge +{bing:.1f}% | spread {spr:+.1f}% | t {t:+.2f} | placebo p {p:.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected.** The spread is **{R['spread']}%** (*t* {R['hedge_t']}) "
            f"— the wrong sign, and the placebo *p* = {R['placebo_p']} says the gap is pure noise. The "
            "bingers slightly out-earned the harvesters."
        ),
        md(
            "### 4c · Robustness — H₃ (any window with a signal?)\n\n"
            "The IC and hedge over four forward/fiscal windows."
        ),
        code(
            "wins = " + repr(R["win"]) + "\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_prices(fetch=False); cf = data.fetch_cashflow(fetch=False)\n"
            "    spans = [('2025-06-27','2026-06-26',2024),('2025-06-27','2025-12-27',2024),\n"
            "             ('2025-01-02','2026-01-02',2024),('2026-02-02','2026-06-26',2025)]\n"
            "    rows = []\n"
            "    for (lab,_,_,_),(sp,en,uy) in zip(wins, spans):\n"
            "        pn = st.build_panel(px, cf, split_date=sp, end_date=en, upto_year=uy)\n"
            "        ic = st.information_coefficient(pn); bb = st.bucket_returns(pn, 0.3)\n"
            "        rows.append((lab, ic['ic'], ic['ic_t'], bb['spread']*100))\n"
            "else:\n"
            "    rows = wins\n"
            "tab = pd.DataFrame(rows, columns=['window','IC','IC_t','spread%']).set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(8.7, 4.3))\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1, label='|t|=2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.bar(range(len(tab)), tab['IC_t'], color=[GREEN if v<-2 else GREY for v in tab['IC_t']], width=.5)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels([w[:12] for w in tab.index], rotation=15)\n"
            "ax.set_ylabel('IC t-stat'); ax.set_ylim(-3, 3)\n"
            "ax.set_title('No window clears -2 in the predicted (negative-IC) direction'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected.** Every window's IC-*t* sits inside ±1.2 and the hedge "
            "spread flips between small negatives — never a *t* ≥ 2 in the predicted direction. A "
            "signal this weak and unstable is not a signal."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (IC +{R['ic']}, *t* +{R['ic_t']}); H₂ rejected "
            f"(spread {R['spread']}%, *t* {R['hedge_t']}, wrong sign); H₃ rejected (unstable). The "
            "yfinance snapshot also caps it below REAL regardless.\n"
            f"- **Tradability `MIRAGE`** — survivor snapshot, binge-leg borrow; gross {R['spread']}% → "
            f"net {R['net']}%.\n"
            "- **Capex cousin of asset growth? `BUSTED`** — vanishes like 244 and 523."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "There's no gross edge; the borrow is a footnote."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.bucket_returns(PANEL, 0.3)\n"
            "    gross = b['spread']*100\n"
            "    net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=100.0, holding_years=1.0)*100\n"
            "else:\n"
            f"    gross, net = {R['spread']}, {R['net']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('harvest - binge spread %')\n"
            "ax.set_title(f'Wrong sign before costs ({gross:+.0f}%), worse after ({net:+.0f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}% -> net {net:+.1f}% (5 bps/leg + 100 bps borrow, 1y hold)')"
        ),
        md(
            "> 💡 In plain words: there is nothing to charge costs against — the "
            "long-harvest/short-binge book *loses* before frictions on this window. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real capex-binge "
            "penalty (`binge_alpha < 0`) of increasing strength and watch the IC-*t* go negative — "
            "averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "alphas = [0.0, -0.10, -0.20, -0.30, -0.40, -0.50]\n"
            "ts = [st.synthetic_mean_ic_t(data, binge_alpha=a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (anomaly)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted binge_alpha (more negative = stronger anomaly)')\n"
            "ax.set_ylabel('mean IC t-stat (25 seeds)')\n"
            "ax.set_title('The harness banks a planted effect — flat at the null, past -2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'binge_alpha {a:+.2f} -> mean IC-t {t:+.2f}')"
        ),
        md(
            "The IC-*t* is ≈ 0 at the null and falls past −2 as the planted effect grows — so the "
            "engine is a faithful detector. The real-tape zero is therefore a statement about **this "
            "survivor snapshot on this window**: the capex-cycle anomaly is absent (faint wrong sign) "
            "and sign-unstable here. For the neighbours it sits between, see "
            "[244 Asset-Growth](../../244-asset-growth/) and "
            "[523 Investment-To-Assets](../../523-investment-to-assets/)."
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
