"""Generate the two narrative notebooks for Study 556 (Electricity-Demand).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached prices under
../_cache/ (+ the always-available hardcoded demand series) if present and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; 234 months
# 2007-2026; demand fp 625867832b4d, panel fp e7d78ee9af2d; signal lagged 2 months).
R = dict(
    fp_dem="625867832b4d", fp_px="dc0f80c14e47", fp_panel="e7d78ee9af2d",
    n_months=234, start="2007-01", end="2026-06",
    # SPY conditional split (1m) + 12m slope
    spy_spread=0.21, spy_t=0.37, spy_p=0.71, spy_slope12_t=-2.77,
    # XLU conditional split (1m, clean non-overlapping) + placebo
    xlu_spread=1.19, xlu_t=2.13, xlu_p=0.032, xlu_slope1_t=1.54,
    # ex-COVID collapse
    xlu_spread_ex=0.54, xlu_t_ex=0.96, xlu_p_ex=0.34, xlu_slope1_t_ex=0.88,
    # overlay
    spy_bh=10.8, spy_ov_net=6.1, xlu_bh=8.5, xlu_ov_gross=3.6, xlu_ov_net=3.5,
    spy_bh_sh=0.74, spy_ov_sh=0.64, xlu_bh_sh=0.63, xlu_ov_sh=0.38, switches=56,
    # predictive regression grid: (tape, horizon, slope_t) for the shown rows
    reg=[("SPY", 1, -0.23), ("SPY", 6, -1.43), ("SPY", 12, -2.77),
         ("XLU", 1, 1.54), ("XLU", 6, 1.42), ("XLU", 12, 0.51)],
    # synthetic control: (edge, mean slope-t over 25 seeds)
    ctrl=[(0.00, 0.37), (0.01, 1.87), (0.02, 3.36), (0.03, 4.81), (0.04, 6.21)],
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

from electricity_demand import data, strategy as st

def load_real():
    \"\"\"Cache-first real monthly panel (empty frame offline).\"\"\"
    return data.load_real()

PANEL = load_real()
HAVE_REAL = not PANEL.empty
print("real demand+equity panel present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(PANEL)} months, {PANEL.index.min().date()} -> {PANEL.index.max().date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Electricity Demand — is the power the economy burns a signal for stocks? ⚡\n"
            "### The grid load as a hard-data pulse of the real economy, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Here's a seductive idea. Almost *everything* the economy does runs on electricity — "
            "factories, data centres, office towers, air conditioners. So the amount of power the "
            "country burns each month should be a **hard, un-fakeable pulse of real activity**: when "
            "the economy is roaring, the meters spin faster. If that's true, growth in electricity "
            "demand might *lead* the stock market — a crystal ball made of kilowatt-hours.\n\n"
            "We test it the honest way: take the U.S. monthly electricity-generation figures, compute "
            "how fast demand is *growing* (versus a year earlier, to cancel out summer/winter), and "
            "check whether fast-growth months are followed by stronger stock returns — for the broad "
            "market (**SPY**) and for the utilities sector (**XLU**), the stocks most directly tied to "
            "the grid.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does fast power-demand growth lead the **broad market**? | **No.** SPY returns after "
            f"hot-demand months are essentially identical to normal (spread +{R['spy_spread']}%, a "
            "coin-flip). |\n"
            f"| Does it lead **utilities** (XLU)? | **A little — but fragile.** XLU earned "
            f"+{R['xlu_spread']}%/mo more after hot months (looks real)… |\n"
            f"| …is that real? | **Mostly it's just 2020.** Drop the COVID year and the edge collapses "
            f"(from +{R['xlu_spread']}% to +{R['xlu_spread_ex']}%/mo). One crash, not a signal. |\n"
            f"| Could you trade it? | **No.** A 'be invested only when demand is hot' rule *loses* to "
            f"just buying and holding — SPY +{R['spy_ov_net']}% vs +{R['spy_bh']}%/yr. |\n\n"
            "> Electricity demand is a real pulse — but a **coincident** one. It moves *with* the "
            "economy, not ahead of it, and the market has already priced whatever the meters are "
            "telling you."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The electricity a country burns is the pulse of its economy — so demand growth should "
            "lead the market.\"*\n\n"
            "The intuition is genuinely good. You can fake a survey; you can't fake a kilowatt-hour. "
            "When a chip fab runs a third shift or a new data centre lights up, it shows up in the "
            "grid load *immediately* and *honestly*. So electricity demand is one of the cleanest "
            "hard-data nowcasts of real activity there is. The leap this study tests is the next one: "
            "does that pulse arrive *early* enough, and un-priced enough, to forecast **stock "
            "returns**?"
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If demand growth led the market, you'd have a macro crystal ball updated monthly from an "
            "official government feed — get defensive when the grid cools, lean in when it heats up. "
            "The desk has tested the same *hard-data-pulse-as-leading-indicator* idea on jobless "
            "claims ([385](../../385-jobless-claims-momentum/)) and shipping rates "
            "([269](../../269-baltic-dry/)); both turned out to be **coincident echoes**, not leaders. "
            "Electricity is the strongest-sounding one yet. Let's look."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure demand growth.** Take monthly U.S. electricity generation and compute the "
            "**year-over-year % change** — comparing July to last July cancels the huge "
            "air-conditioning/heating seasonal and leaves the business-cycle pulse.\n"
            "2. **Call it 'hot' or 'cold'.** Hot = growth above its recent typical level.\n"
            "3. **Wait for the data to be public.** The government publishes each month's figure "
            "~2 months late, so we only ever act on data we *could* have had (a 2-month lag).\n"
            "4. **Compare forward returns.** Did stocks do better after hot-demand months than cold "
            "ones? The believers say yes.\n\n"
            "*No peeking:* the signal uses only demand data that was already published, and we test "
            "the return that comes *after*."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Broad market first: do hot-demand months lead to better SPY returns?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig = st.signal_lagged(PANEL['netgen'])\n"
            "    out = {}\n"
            "    for tk in ['SPY','XLU']:\n"
            "        cs = st.conditional_split(sig, PANEL[tk], horizon=1)\n"
            "        out[tk] = (cs['hot_mean']*100, cs['cold_mean']*100, cs['spread']*100, cs['t'])\n"
            "    banner = 'REAL tape (234 months, 2007-2026)'\n"
            "else:\n"
            f"    out = {{'SPY': (1.08, 0.86, {R['spy_spread']}, {R['spy_t']}),\n"
            f"           'XLU': (1.44, 0.26, {R['xlu_spread']}, {R['xlu_t']})}}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)\n"
            "for ax, tk in zip(axes, ['SPY','XLU']):\n"
            "    hot, cold, spr, t = out[tk]\n"
            "    ax.bar(['after HOT\\ndemand','after COLD\\ndemand'], [hot, cold], color=[GREEN, GREY], width=.55)\n"
            "    ax.axhline(0, c='k', lw=1); ax.set_title(f'{tk}: spread {spr:+.2f}%/mo (t {t:+.2f})')\n"
            "axes[0].set_ylabel('avg next-month return %')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "for tk in ['SPY','XLU']:\n"
            "    hot, cold, spr, t = out[tk]; print(f'{tk}: hot +{hot:.2f}% cold +{cold:.2f}% -> spread {spr:+.2f}%/mo (t {t:+.2f})')"
        ),
        md(
            f"For **SPY the answer is basically nothing** — hot and cold months look the same "
            f"(spread +{R['spy_spread']}%, *t* {R['spy_t']}). For **XLU (utilities)** there's a real "
            f"gap: +{R['xlu_spread']}%/mo more after hot demand (*t* {R['xlu_t']}). Utilities *do* seem "
            "to respond to the grid pulse. But before we get excited — is that a repeatable pattern, "
            "or one big moment?"
        ),
        md(
            "**The honesty check: what happens if we remove 2020?** In 2020 the lockdown crushed "
            "electricity demand *and* the stock market at the same time — a single dramatic event that "
            "can look like 'demand predicts returns'."
        ),
        code(
            "labels = ['full sample\\n(2007-2026)', 'drop 2020\\n(ex-COVID)']\n"
            "if HAVE_REAL:\n"
            "    sig = st.signal_lagged(PANEL['netgen'])\n"
            "    full = st.conditional_split(sig, PANEL['XLU'], 1)\n"
            "    m = ~((PANEL.index >= '2020-01-01') & (PANEL.index <= '2020-12-31'))\n"
            "    px = PANEL[m]; sx = st.signal_lagged(px['netgen'])\n"
            "    ex = st.conditional_split(sx, px['XLU'], 1)\n"
            "    spreads = [full['spread']*100, ex['spread']*100]; ts = [full['t'], ex['t']]\n"
            "else:\n"
            f"    spreads = [{R['xlu_spread']}, {R['xlu_spread_ex']}]; ts = [{R['xlu_t']}, {R['xlu_t_ex']}]\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "cols = [GREEN if t>2 else RED for t in ts]\n"
            "bars = ax.bar(labels, spreads, color=cols, width=.5)\n"
            "for b, t in zip(bars, ts): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f't {t:+.2f}', ha='center')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('XLU hot-minus-cold spread %/mo')\n"
            "ax.set_title('The XLU edge is (mostly) the COVID crash — remove 2020 and it fades')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'XLU spread full {spreads[0]:+.2f}% (t {ts[0]:+.2f}) -> ex-2020 {spreads[1]:+.2f}% (t {ts[1]:+.2f})')"
        ),
        md(
            f"There it is. Drop one year — 2020 — and the utilities 'signal' falls from "
            f"+{R['xlu_spread']}%/mo (*t* {R['xlu_t']}, looked real) to +{R['xlu_spread_ex']}%/mo "
            f"(*t* {R['xlu_t_ex']}, gone). The edge was **one macro event**, not a repeatable pulse."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The broad market shows no forward signal at all; utilities show one "
            "borderline reading that is essentially the 2020 crash and dies when you remove it.\n"
            "- **Tradability — Mirage.** Turning even the utilities pattern into a 'be invested only "
            "when demand is hot' rule *loses* to buy-and-hold (next beat)."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The natural trade: hold the ETF when demand growth is hot, sit in cash when it's cold. "
            "Does it beat just buying and holding?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig = st.signal_lagged(PANEL['netgen'])\n"
            "    res = {tk: st.overlay_backtest(sig, PANEL[tk]) for tk in ['SPY','XLU']}\n"
            "    bh = [res['SPY']['bh_ann']*100, res['XLU']['bh_ann']*100]\n"
            "    ov = [res['SPY']['overlay_net_ann']*100, res['XLU']['overlay_net_ann']*100]\n"
            "else:\n"
            f"    bh = [{R['spy_bh']}, {R['xlu_bh']}]; ov = [{R['spy_ov_net']}, {R['xlu_ov_net']}]\n"
            "x = np.arange(2); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(x-w/2, bh, w, label='buy & hold', color=GREEN)\n"
            "ax.bar(x+w/2, ov, w, label='hold-when-hot (net)', color=RED)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['SPY','XLU']); ax.set_ylabel('annualised return %')\n"
            "ax.set_title('Acting on the demand pulse LOSES to buy-and-hold'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for i, tk in enumerate(['SPY','XLU']): print(f'{tk}: buy-hold +{bh[i]:.1f}%/yr vs overlay +{ov[i]:.1f}%/yr')"
        ),
        md(
            "The overlay underperforms on both tapes — you sit in cash through too much of the rally. "
            "Even where the raw pattern looked real (XLU), as a *position* it destroys return. "
            "`MIRAGE`."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Same trap, other pulses.** [385 Jobless-Claims-Momentum](../../385-jobless-claims-momentum/) "
            "and [269 Baltic-Dry](../../269-baltic-dry/) test the identical 'hard-data leads the market' "
            "idea on claims and shipping — both land coincident, not leading.\n"
            "- **The real test needs higher frequency and cleaner attribution.** Weekly/daily regional "
            "grid load, split by industrial vs residential, might isolate the *industrial* pulse that "
            "actually leads — beyond a monthly national total dominated by weather.\n\n"
            "*Think the pulse is real and we just averaged it away? Fork this, pull weekly EIA-930 "
            "balancing-authority load, and show a demand-growth signal clearing t = 2 the right way "
            "ex-COVID.*"
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
            "# Electricity Demand — a quantitative teardown 🔬\n"
            "### YoY demand-growth momentum · HAC predictive regression · hot-vs-cold Welch *t* · label-shuffle placebo · ex-COVID robustness · overlay vs buy-and-hold · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build a year-over-year "
            "electricity-demand-growth signal, lag it two months so it is strictly public, and test "
            "whether it forecasts SPY and XLU returns.\n\n"
            "> ⚠️ **Not investment advice.** Real data: a hardcoded monthly snapshot of EIA net "
            f"generation (`ELEC.GEN.ALL-US-99.M`) + yfinance SPY/XLU, 234 months (as-of 2026-06-30, "
            f"panel fp `{R['fp_panel']}`); the offline core runs on a deterministic synthetic world. "
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
            f"| **Signal** | `WEAK` | SPY hot-cold spread **+{R['spy_spread']}%** (*t* **{R['spy_t']}**, "
            f"*p* {R['spy_p']}) — nothing; its only \\|*t*\\| ≥ 2 (12m slope **{R['spy_slope12_t']}**) is "
            f"the *wrong* sign. XLU 1m spread **+{R['xlu_spread']}%** (*t* **{R['xlu_t']}**, *p* "
            f"{R['xlu_p']}) clears the bar but **dies ex-2020** (*t* {R['xlu_t_ex']}, *p* {R['xlu_p_ex']}). |\n"
            f"| **Tradability** | `MIRAGE` | Hold-when-hot overlay < buy-and-hold on both tapes "
            f"(SPY +{R['spy_ov_net']}% vs +{R['spy_bh']}%/yr; XLU Sharpe {R['xlu_ov_sh']} vs {R['xlu_bh_sh']}). |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on this tape "
            "electricity demand is a coincident, seasonal, mostly-priced pulse — one COVID-driven "
            "utilities reading and nothing else."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $g_t$ be lagged demand-growth (YoY % change of net generation, shifted 2 months so "
            "it is public) and $r_{t\\to t+h}$ the forward $h$-month return.\n\n"
            "- **H₁ (predictive slope).** In $r_{t\\to t+h} = \\alpha + \\beta g_t + \\varepsilon$, "
            "$\\beta > 0$ at HAC *t* ≥ 2 (hot demand → higher forward returns).\n"
            "- **H₂ (conditional split).** $\\mathbb{E}[r\\mid\\text{hot}] - "
            "\\mathbb{E}[r\\mid\\text{cold}] > 0$ at *t* ≥ 2.\n"
            "- **H₃ (robustness).** H₁/H₂ survive dropping the COVID-2020 year.\n"
            "- **H₄ (tradable).** A hold-when-hot overlay beats buy-and-hold net of costs.\n\n"
            "We find **H₁ rejected** (SPY wrong-sign, XLU sub-2), **H₂ borderline on XLU only** "
            "(*t* +2.13), **H₃ rejected** (the XLU reading is a COVID artefact), and **H₄ rejected** "
            "(the overlay loses)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Electricity demand is a superb *coincident* nowcast — it "
            "co-moves with GDP almost mechanically. But two things blunt it as a *forward* equity "
            "signal: (a) the informative part is a small residual once you strip the enormous "
            "summer/winter seasonal (the YoY transform), and (b) whatever activity the grid reveals "
            "is largely *already priced* by the time the monthly figure prints ~8 weeks late. That is "
            "the same coincident-echo verdict the desk reached for "
            "[385 jobless claims](../../385-jobless-claims-momentum/) and "
            "[269 the Baltic Dry index](../../269-baltic-dry/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $g_t = $ YoY % change of monthly net generation, then `shift(2)` (1 month "
            "for the EIA publication lag, 1 for the signal→return convention). 'Hot' = above the "
            "trailing 24-month median.\n"
            "- **Predictive regression.** OLS of forward $h$-month return on $g_t$, **Newey-West** "
            "HAC standard errors with lags = $h$ (overlap correction).\n"
            "- **Conditional split.** Welch two-sample *t* on hot vs cold forward returns; a "
            "**label-shuffle placebo** null (2000+ perms).\n"
            "- **Robustness.** Re-run ex-2020; scan horizons 1/3/6/12m and both tapes.\n"
            "- **Overlay.** Hold-when-hot vs buy-and-hold, 5 bps per switch.\n"
            "- **Positive control.** A deterministic synthetic world with a planted demand→returns "
            "link (`edge > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Inference note: only the **1-month** horizon gives *non-overlapping* returns, so it is the "
            "clean two-sample test; the 3/6/12m splits overlap and their *t* is inflated — shown for "
            "shape, used for HAC regression only."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — H₁\n\n"
            "Forward return on lagged demand growth, HAC *t* on the slope, across horizons and both "
            "tapes. Believers' sign is positive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig = st.signal_lagged(PANEL['netgen'])\n"
            "    rows = []\n"
            "    for tk in ['SPY','XLU']:\n"
            "        for h in [1,3,6,12]:\n"
            "            reg = st.predictive_regression(sig, PANEL[tk], horizon=h)\n"
            "            rows.append((tk, h, reg['slope_t']))\n"
            "    grid = pd.DataFrame(rows, columns=['tape','h','slope_t'])\n"
            "else:\n"
            f"    grid = pd.DataFrame({R['reg']!r}, columns=['tape','h','slope_t'])\n"
            "piv = grid.pivot(index='h', columns='tape', values='slope_t')\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "for tk, c in [('SPY', GREY), ('XLU', GREEN)]:\n"
            "    if tk in piv: ax.plot(piv.index, piv[tk], 'o-', c=c, lw=2, label=tk)\n"
            "ax.axhline(2, ls='--', c=AMBER, lw=1); ax.axhline(-2, ls='--', c=AMBER, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('forecast horizon (months)')\n"
            "ax.set_ylabel('HAC slope-t'); ax.set_title('The only |t|>=2 is SPY at 12m — and it is NEGATIVE (wrong sign)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(piv.round(2).to_string())"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected.** XLU leans positive but never clears the HAC bar "
            f"(best *t* +{R['xlu_slope1_t']}). The one slope past |*t*| = 2 is **SPY at 12 months, "
            f"*t* {R['spy_slope12_t']}** — the *wrong* sign (fast demand growth → *weaker* broad market, "
            "the late-cycle overheating read), on the most overlap-inflated horizon."
        ),
        md(
            "### 4b · The conditional split — H₂ (clean 1-month test)\n\n"
            "Hot vs cold forward 1-month returns, Welch *t* + label-shuffle placebo."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig = st.signal_lagged(PANEL['netgen'])\n"
            "    out = {}\n"
            "    for tk in ['SPY','XLU']:\n"
            "        cs = st.conditional_split(sig, PANEL[tk], 1)\n"
            "        p = st.placebo_pvalue(sig, PANEL[tk], 1, n_perm=2000, seed=556)\n"
            "        out[tk] = (cs['spread']*100, cs['t'], p)\n"
            "else:\n"
            f"    out = {{'SPY': ({R['spy_spread']}, {R['spy_t']}, {R['spy_p']}),\n"
            f"           'XLU': ({R['xlu_spread']}, {R['xlu_t']}, {R['xlu_p']})}}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "tks = ['SPY','XLU']; spreads = [out[t][0] for t in tks]; ts = [out[t][1] for t in tks]\n"
            "cols = [GREEN if t>2 else GREY for t in ts]\n"
            "bars = ax.bar(tks, spreads, color=cols, width=.5)\n"
            "for b, t, tk in zip(bars, ts, tks): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f't {t:+.2f}\\np {out[tk][2]:.3f}', ha='center')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('hot-minus-cold spread %/mo')\n"
            "ax.set_title('Only XLU clears t=2 (barely) — SPY is a coin flip'); plt.tight_layout(); plt.show()\n"
            "for tk in tks: print(f'{tk}: spread {out[tk][0]:+.2f}%/mo, welch t {out[tk][1]:+.2f}, placebo p {out[tk][2]:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ passes **only for XLU, and only just** (spread +{R['xlu_spread']}%, "
            f"*t* +{R['xlu_t']}, placebo *p* {R['xlu_p']}). SPY is a coin flip (*t* +{R['spy_t']}). Hold "
            "that XLU reading up to the robustness light next."
        ),
        md(
            "### 4c · Robustness — H₃ (does the XLU reading survive ex-COVID?)\n\n"
            "The one real-looking reading, with and without calendar-2020."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig = st.signal_lagged(PANEL['netgen'])\n"
            "    full = st.conditional_split(sig, PANEL['XLU'], 1)\n"
            "    pf = st.placebo_pvalue(sig, PANEL['XLU'], 1, n_perm=2000, seed=556)\n"
            "    m = ~((PANEL.index >= '2020-01-01') & (PANEL.index <= '2020-12-31'))\n"
            "    px = PANEL[m]; sx = st.signal_lagged(px['netgen'])\n"
            "    ex = st.conditional_split(sx, px['XLU'], 1)\n"
            "    pe = st.placebo_pvalue(sx, px['XLU'], 1, n_perm=2000, seed=556)\n"
            "    spreads = [full['spread']*100, ex['spread']*100]; ts = [full['t'], ex['t']]; ps = [pf, pe]\n"
            "else:\n"
            f"    spreads = [{R['xlu_spread']}, {R['xlu_spread_ex']}]; ts = [{R['xlu_t']}, {R['xlu_t_ex']}]; ps = [{R['xlu_p']}, {R['xlu_p_ex']}]\n"
            "labels = ['full (2007-26)','ex-2020']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "cols = [GREEN if t>2 else RED for t in ts]\n"
            "bars = ax.bar(labels, spreads, color=cols, width=.5)\n"
            "for b, t, p in zip(bars, ts, ps): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f't {t:+.2f}\\np {p:.3f}', ha='center')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('XLU hot-minus-cold spread %/mo')\n"
            "ax.set_title('H3 rejected: the XLU edge is a COVID artefact'); plt.tight_layout(); plt.show()\n"
            "print(f'XLU 1m: full {spreads[0]:+.2f}% (t {ts[0]:+.2f}, p {ps[0]:.3f}) -> ex-2020 {spreads[1]:+.2f}% (t {ts[1]:+.2f}, p {ps[1]:.3f})')"
        ),
        md(
            f"> 💡 In plain words: H₃ **rejected.** Remove 2020 and the XLU spread falls from "
            f"+{R['xlu_spread']}% (*t* +{R['xlu_t']}, *p* {R['xlu_p']}) to +{R['xlu_spread_ex']}% "
            f"(*t* +{R['xlu_t_ex']}, *p* {R['xlu_p_ex']}). The entire edge is the COVID demand-and-market "
            "crash — one event, not a repeatable pulse."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁ rejected (SPY wrong-sign 12m slope {R['spy_slope12_t']}, XLU "
            f"sub-2); H₂ passes on XLU only (*t* +{R['xlu_t']}); H₃ rejected (COVID artefact, ex-2020 "
            f"*t* +{R['xlu_t_ex']}). Literature-plausible, one fragile reading — `WEAK`, not `REAL`.\n"
            f"- **Tradability `MIRAGE`** — the overlay loses to buy-and-hold on both tapes.\n"
            "- No `REAL`: a real-tape HAC *t* ≥ 2 the *right* way, robust ex-COVID, never appears."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the overlay\n\n"
            "Hold the ETF when demand is hot, else cash; 5 bps per switch."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sig = st.signal_lagged(PANEL['netgen'])\n"
            "    res = {tk: st.overlay_backtest(sig, PANEL[tk]) for tk in ['SPY','XLU']}\n"
            "    bh = [res[t]['bh_ann']*100 for t in ['SPY','XLU']]\n"
            "    ov = [res[t]['overlay_net_ann']*100 for t in ['SPY','XLU']]\n"
            "    shb = [res[t]['bh_sharpe'] for t in ['SPY','XLU']]; sho = [res[t]['overlay_sharpe'] for t in ['SPY','XLU']]\n"
            "else:\n"
            f"    bh = [{R['spy_bh']}, {R['xlu_bh']}]; ov = [{R['spy_ov_net']}, {R['xlu_ov_net']}]\n"
            f"    shb = [{R['spy_bh_sh']}, {R['xlu_bh_sh']}]; sho = [{R['spy_ov_sh']}, {R['xlu_ov_sh']}]\n"
            "x = np.arange(2); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(x-w/2, bh, w, label='buy & hold', color=GREEN)\n"
            "ax.bar(x+w/2, ov, w, label='overlay (net)', color=RED)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['SPY','XLU']); ax.set_ylabel('annualised return %')\n"
            "ax.set_title('Overlay loses to buy-and-hold, gross and net'); ax.legend(); plt.tight_layout(); plt.show()\n"
            "for i, tk in enumerate(['SPY','XLU']): print(f'{tk}: BH +{bh[i]:.1f}%/yr (Sharpe {shb[i]:.2f}) vs overlay +{ov[i]:.1f}%/yr (Sharpe {sho[i]:.2f})')"
        ),
        md(
            "> 💡 In plain words: even the XLU reading, once it becomes a *position*, destroys return "
            f"(Sharpe {R['xlu_ov_sh']} vs {R['xlu_bh_sh']} buy-and-hold). There is nothing to allocate "
            "to. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real "
            "demand→returns link (`edge > 0`) of increasing strength and watch the HAC slope-*t* rise "
            "— averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "edges = [0.0, 0.01, 0.02, 0.03, 0.04]\n"
            "ts = [st.synthetic_mean_t(data, edge=e, n_seeds=25) for e in edges]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(edges, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=AMBER, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted edge (stronger demand->returns link)')\n"
            "ax.set_ylabel('mean HAC slope-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted pulse — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e, t in zip(edges, ts): print(f'edge {e:+.2f} -> mean slope-t {t:+.2f}')"
        ),
        md(
            "The slope-*t* is ≈ 0 at the null and rises past +2 as the planted pulse grows — so the "
            "engine is a faithful detector. The real-tape result is therefore a statement about **the "
            "tape, not the code**: U.S. electricity demand is a coincident, seasonal, mostly-priced "
            "pulse whose only real-looking equity reading is a COVID artefact. For the sibling "
            "hard-data-pulse studies, see [385 Jobless-Claims-Momentum](../../385-jobless-claims-momentum/) "
            "and [269 Baltic-Dry](../../269-baltic-dry/)."
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
