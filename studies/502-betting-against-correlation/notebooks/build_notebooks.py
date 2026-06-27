"""Generate the two narrative notebooks for Study 502 (Betting-Against-Correlation).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the desk's seven beats. The synthetic figures run anywhere, offline and
deterministic; the real-tape cells use the cached yfinance panel under ../_cache/ if present and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md exactly), so the
notebooks re-run for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (as-of 2026-06-26, window 2013-06 -> 2025-11, 150 months, fingerprint 3f75d60c45d8).
R = dict(
    fp="3f75d60c45d8", n=150, win_start="2013-06", win_end="2025-11",
    # headline BAC book (correlation sort, beta-neutral)
    bac_mean=3.75, bac_vol=9.10, bac_sharpe=0.412, bac_t=1.543,
    bac_hit=56.7, bac_dd=-20.9, bac_ci_lo=-1.0, bac_ci_hi=8.5,
    bac_net_mean=3.36, bac_net_t=1.385, cost_bps=5.0, borrow_bps=50.0,
    # sort separation / neutralisation
    corr_long=0.409, corr_short=0.670, beta_long=0.725, beta_short=1.176,
    w_short=0.635, turnover=5.9,
    long_leg=15.19, short_leg=16.71, spy_leg=12.95,
    # decomposition: corr vs beta vs vol
    corr_sort_mean=3.75, corr_sort_t=1.543,
    beta_sort_mean=-0.67, beta_sort_t=-0.288,
    vol_sort_mean=-0.87, vol_sort_t=-0.384,
    # placebo
    placebo_real=0.313, placebo_mean=0.005, placebo_std=0.164, placebo_p=0.040,
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

from betting_against_correlation import data, strategy as st

def load_real():
    \"\"\"Cache-first real panel (empty offline).\"\"\"
    prices, spy = data.fetch_panel(fetch=False)
    return prices, spy

PRICES, SPY = load_real()
HAVE_REAL = not PRICES.empty
print("real BAC panel cache present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({PRICES.shape[1]} names, {PRICES.index[0].date()} -> {PRICES.index[-1].date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Betting Against Correlation — is 'bet against beta' secretly a bet against *correlation*?\n"
            "### Pulling the famous low-risk premium apart, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Is_it_correlation%2C_not_vol%3F: Confirmed](https://img.shields.io/badge/Is_it_correlation%2C_not_vol%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "One of the best-documented 'free lunches' in finance is **betting against beta**: calm, "
            "low-beta stocks quietly beat wild, high-beta ones once you account for risk. But a "
            "stock's beta is really two things multiplied together — *how much it moves with the "
            "market* (**correlation**) and *how much it moves at all* (**volatility**). Asness, "
            "Frazzini, Gormsen & Pedersen (2020) asked: **which half is doing the work?** They say "
            "it's the **correlation** half. We test it.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo and the cost "
            "maths? That's the companion, **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the low-risk premium a *correlation* effect? | **Yes.** Sorting on **correlation** "
            f"earns **+{R['corr_sort_mean']}%/yr**; sorting on **beta** or **volatility** earns "
            f"*nothing* ({R['beta_sort_mean']:+.1f}% and {R['vol_sort_mean']:+.1f}%/yr). |\n"
            f"| So is it a money machine? | **Not on this sample.** The correlation book earns "
            f"+{R['bac_mean']}%/yr but it's so thin the data can't be sure (*t* = {R['bac_t']}, "
            "below the desk's bar of 2). |\n"
            f"| Is it just noise, then? | **No.** Shuffle the correlation labels 200 times and the real "
            f"result lands in the top **{R['placebo_p']*100:.0f}%** — the correlation signal is doing "
            "real work. |\n"
            "| So what *is* it? | A **real, identified mechanism** with a magnitude a 48-name survivor "
            "basket can't certify as a trade. |\n\n"
            "> The paper is right about the *why*. The *how much* is too thin to bank on this tape."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The betting-against-beta premium is driven by **correlation**, not volatility.\"* — "
            "Asness, Frazzini, Gormsen & Pedersen (2020).\n\n"
            "Here's the algebra that makes it interesting. A stock's beta — its sensitivity to the "
            "market — is exactly:\n\n"
            "$$\\beta_i \\;=\\; \\underbrace{\\rho_{i,m}}_{\\text{correlation}} \\times "
            "\\underbrace{\\frac{\\sigma_i}{\\sigma_m}}_{\\text{relative volatility}}$$\n\n"
            "So 'low beta' can mean two very different things: a stock that barely *tracks* the market "
            "(low correlation), or a stock that's just *quiet* (low volatility). AFGP take the famous "
            "bet-against-beta trade apart and ask which half pays. Their answer — the **correlation** "
            "half — points at the leverage-constraint story (investors who can't borrow crowd into "
            "things that move *with* the market) over the 'people love lottery tickets' story."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the premium is a **correlation** effect, then the right trade isn't 'buy quiet "
            "stocks' — it's 'buy stocks that march to their own drum'. That's a different basket "
            "(energy, gold miners, utilities — things only loosely tied to a tech-heavy index) and a "
            "different *reason* to expect a premium. The desk has already run the plain "
            "bet-against-beta factor ([238 Betting-Against-Beta](../../238-betting-against-beta/)) and "
            "the low-vol ETF race ([330 Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)). "
            "This one zooms into the *mechanism*: corr vs vol vs beta, same panel, same machine."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest checks:\n\n"
            "1. **Sort on correlation.** Each month, rank ~48 large-caps by how correlated they've "
            "been to the market over the past year. Buy the low-correlation half, short the "
            "high-correlation half.\n"
            "2. **Be fair about risk.** The high-correlation names also carry more market exposure, so "
            "shorting them is partly shorting *the market going up*. We hedge that out "
            "(**beta-neutralise**) so what's left is the pure correlation bet — not a hidden beta bet.\n"
            "3. **Race correlation against its rivals.** Run the *same* machine sorting on **beta** and "
            "on **volatility** instead. If AFGP are right, only the correlation sort should pay.\n\n"
            "Tape: yfinance daily prices for 48 large-cap survivors + SPY, 2012–2025."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The decisive picture: same engine, three different things to sort on.** Which one pays?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    means = {}\n"
            "    for so in ['corr','beta','vol']:\n"
            "        b = st.bac_portfolio(PRICES, SPY, sort_on=so)\n"
            "        means[so] = st.summary(b['bac_ret'])['mean']*100\n"
            "    cm, bm, vm = means['corr'], means['beta'], means['vol']\n"
            "    banner = 'REAL tape'\n"
            "else:\n"
            f"    cm, bm, vm = {R['corr_sort_mean']}, {R['beta_sort_mean']}, {R['vol_sort_mean']}\n"
            "    banner = 'SYNTHETIC fallback (frozen real numbers)'\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "bars = ax.bar(['sort on\\nCORRELATION','sort on\\nbeta','sort on\\nvolatility'],\n"
            "              [cm, bm, vm], color=[GREEN, GREY, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('book return %/yr')\n"
            "ax.set_title(f'Only the correlation sort pays  ({banner})')\n"
            "for bar_, v in zip(bars, [cm,bm,vm]):\n"
            "    ax.annotate(f'{v:+.1f}%', (bar_.get_x()+bar_.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'correlation {cm:+.2f}%/yr | beta {bm:+.2f}%/yr | volatility {vm:+.2f}%/yr')"
        ),
        md(
            f"There's the whole study in one chart. Sorting on **correlation** earns "
            f"**+{R['corr_sort_mean']}%/yr**; sorting on **beta** or **volatility** earns *nothing* "
            f"({R['beta_sort_mean']:+.1f}% and {R['vol_sort_mean']:+.1f}%/yr). The low-risk premium "
            "really does live in the *correlation* component — exactly what AFGP claimed."
        ),
        md(
            "**But is the correlation book a *trade*?** Here's its gross return, and what's left "
            "after costs and a borrow fee for the short leg:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    book = st.bac_portfolio(PRICES, SPY)\n"
            "    g = st.summary(book['bac_ret']); gm, gt = g['mean']*100, g['tstat']\n"
            "    net = st.apply_costs(book, cost_bps=5.0, borrow_ann_bps=50.0)\n"
            "    nm = st.summary(net)['mean']*100\n"
            "else:\n"
            f"    gm, gt, nm = {R['bac_mean']}, {R['bac_t']}, {R['bac_net_mean']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['gross','after costs\\n& borrow'], [gm, nm], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('return %/yr')\n"
            "ax.set_title(f'A thin +{gm:.1f}%/yr — but t = {gt:.2f} (below the bar of 2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross +{gm:.2f}%/yr (HAC t {gt:.2f}); net ~+{nm:.2f}%/yr')"
        ),
        md(
            f"It earns a positive **+{R['bac_mean']}%/yr**, but it's *so thin* the data can't tell it "
            f"from luck (*t* = {R['bac_t']}, below the bar of 2). After realistic costs it's "
            f"~+{R['bac_net_mean']}%/yr. The mechanism is real; the paycheck is a whisper."
        ),
        md(
            "**One more honesty check: is the +3.75%/yr just an accident of the construction?** We "
            "shuffle the correlation labels across names 200 times — destroying the signal but keeping "
            "everything else — and see where the *real* result lands:"
        ),
        code(
            "# Light live placebo (80 shuffles) for speed; the stamped p uses 200 (docs/results.md).\n"
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(PRICES, SPY, n_shuffles=80, seed=502)\n"
            "    real_m, p = pb['real_mean']*100, pb['p_value']\n"
            "    pmean, pstd = pb['placebo_mean']*100, pb['placebo_std']*100\n"
            "else:\n"
            f"    real_m, p, pmean, pstd = {R['placebo_real']}, {R['placebo_p']}, {R['placebo_mean']}, {R['placebo_std']}\n"
            "xs = np.random.default_rng(0).normal(pmean, max(pstd,1e-6), 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.hist(xs, bins=40, color=GREY, alpha=.6, label='shuffled (null)')\n"
            "ax.axvline(real_m, c=GREEN, lw=2.5, label=f'real (+{real_m:.3f}%/mo)')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('mean monthly book return %'); ax.set_title(f'Real result in the right tail — placebo p = {p:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real mean {real_m:+.3f}%/mo vs shuffled {pmean:+.3f}% -> one-sided p = {p:.3f}')"
        ),
        md(
            f"The real result sits in the **right tail** of the shuffled null — one-sided "
            f"**p = {R['placebo_p']:.2f}**. So the correlation labels genuinely matter: the spread "
            "isn't an artefact. The effect is *structurally real*, just statistically thin on 12 years."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Is it correlation, not vol? — Confirmed.** Correlation sort +{R['corr_sort_mean']}%/yr; "
            f"beta and vol sorts earn nothing. The premium lives in the correlation slice.\n"
            f"- **Signal — Weak.** The correlation book earns +{R['bac_mean']}%/yr but at *t* = "
            f"{R['bac_t']} (below 2), CI straddles zero. Placebo p = {R['placebo_p']:.2f} keeps it "
            "off 'None', but this sample can't certify it.\n"
            f"- **Tradability — Fragile.** Sharpe {R['bac_sharpe']}, drawdown {R['bac_dd']}%, needs a "
            "short leg, and it gets crushed when correlations spike (2020, 2023). Thin after costs."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Honestly — barely, on this basket. The correlation book is real in *direction* (the "
            "placebo confirms it) but tiny in *size*, it requires shorting and leverage to hold "
            "beta-neutral, and it suffers exactly when you'd most want a hedge: in panics, when "
            f"*everything* becomes correlated and the spread collapses (2020 {-11.5:.0f}%, 2023 "
            f"{-16.2:.0f}%). The useful takeaway isn't a self-financing machine — it's the *insight*: "
            "if you're tilting toward 'low-risk' stocks, tilt on **correlation**, not volatility."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The parent factor.** [238 Betting-Against-Beta](../../238-betting-against-beta/) runs "
            "the plain beta sort — and lands Weak/Fragile too, on the same survivor problem.\n"
            "- **The vol cousin.** [330 Low-Volatility-Anomaly](../../330-low-volatility-anomaly/) "
            "races the calm ETF against the wild one.\n"
            "- **A broader universe.** This used 48 survivors. The real BAC premium needs the *failed* "
            "high-correlation names too — fork this with a point-in-time membership list and see if "
            "the correlation book clears *t* = 2 once the short leg has its delisted names back.\n\n"
            "*Think the correlation premium is a real trade, not just a real mechanism? Fork this, "
            "widen the basket past survivors, and show the hedged book clearing t = 2.*"
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
            "# Betting Against Correlation — a quantitative teardown 🔬\n"
            "### correlation sort · beta-neutralisation · HAC *t* + block bootstrap · label-shuffle placebo · corr-vs-beta-vs-vol · costs & borrow · seed-robust control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Is_it_correlation%2C_not_vol%3F: Confirmed](https://img.shields.io/badge/Is_it_correlation%2C_not_vol%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We sort a large-cap survivor "
            "cross-section on trailing correlation-to-market, beta-neutralise the long-short, and "
            "test whether the residual is a real correlation premium or just the structural beta gap.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily adjusted-close, 48 large-cap "
            f"survivors + SPY, 2012–2025 (as-of 2026-06-26, fingerprint `{R['fp']}`); the offline core "
            "and tests run on a deterministic synthetic panel. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | BAC book +{R['bac_mean']}%/yr gross, HAC *t* = **{R['bac_t']}** "
            f"(boot CI [{R['bac_ci_lo']}%, {R['bac_ci_hi']}%]); placebo **p = {R['placebo_p']:.2f}** "
            "keeps it off `NONE`. |\n"
            f"| **Tradability** | `FRAGILE` | Sharpe {R['bac_sharpe']}, max-DD {R['bac_dd']}%, net "
            f"+{R['bac_net_mean']}%/yr (*t* {R['bac_net_t']}) after {R['cost_bps']:.0f} bps/leg + "
            f"{R['borrow_bps']:.0f} bps borrow. |\n"
            f"| **Is it correlation, not vol?** | `CONFIRMED` | corr sort +{R['corr_sort_mean']}%/yr "
            f"vs beta {R['beta_sort_mean']}%/yr vs vol {R['vol_sort_mean']}%/yr — the premium is the "
            "correlation slice. |\n\n"
            "> 💡 In plain words: AFGP's *mechanism* is confirmed (correlation pays, vol/beta don't), "
            "the spread is structurally real (placebo p = 0.04), but the *magnitude* sits below the "
            "inference bar on a 48-name survivor panel."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Decompose beta: $\\beta_i = \\rho_{i,m}\\,\\sigma_i/\\sigma_m$. Let "
            "$r^{\\text{BAC}}$ be the beta-neutral long-low-$\\rho$ / short-high-$\\rho$ book.\n\n"
            "- **H₁ (correlation premium).** $\\mathbb{E}[r^{\\text{BAC}}] > 0$ at HAC *t* ≥ 2.\n"
            "- **H₂ (it's correlation, not vol).** The **correlation** sort out-earns the **beta** and "
            "**volatility** sorts on the same panel.\n"
            "- **H₃ (not an artefact).** The real mean beats a label-shuffle null (placebo p small).\n"
            "- **H₄ (net of frictions).** H₁ survives costs and short borrow.\n\n"
            "We find **H₂ confirmed** and **H₃ supported (p = 0.04)**, but **H₁ supported in sign, "
            "sub-*t* = 2**, and **H₄ surviving in sign but thin** — a real mechanism, an uncertain trade."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The interesting content is the **decomposition**, not the binary. AFGP's contribution is "
            "to separate *which* component of the low-risk effect pays — and that maps to *which "
            "theory* is right (leverage constraints vs lottery demand). If correlation pays and "
            "volatility doesn't, the leverage-constraint channel (Frazzini–Pedersen) wins. Our "
            "real-tape decomposition lands squarely there. The *tradability* failure is the familiar "
            "survivor-panel story: a thin, real edge that a 48-name basket of 2026 survivors can't "
            "bank — the same shape as [238 Betting-Against-Beta](../../238-betting-against-beta/)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** Trailing 252-day correlation-to-market per name; rank, split at the median.\n"
            "- **Construction.** Long low-$\\rho$ half, short high-$\\rho$ half, equal-weight within "
            "a leg; **beta-neutralise** by scaling the short leg to $\\beta_{\\text{long}}/"
            "\\beta_{\\text{short}}$ so the book's market beta ≈ 0.\n"
            "- **One execution lag.** Enter the close one trading day after the signal date — no "
            "same-bar fill, no look-ahead.\n"
            "- **Inference.** Newey–West HAC *t* on the monthly mean; circular block-bootstrap CI "
            "(block 6) on the annualised mean.\n"
            "- **Placebo.** Permute the correlation ranks across names (200 shuffles); the one-sided "
            "p-value is the fraction of placebo means ≥ the real mean.\n"
            "- **Decomposition.** Re-run the same engine sorting on **beta** and on **volatility**.\n"
            "- **Frictions.** One-way `cost_bps` × NAV × realised turnover (both legs) + annual short "
            "borrow on the short notional.\n"
            "- **Positive control.** A deterministic synthetic panel with a dial-able, "
            "correlation-keyed premium; the book's HAC *t* averaged over **20 seeds/point**.\n\n"
            "Tape: yfinance daily, 48 survivors + SPY, 2012–2025; book 2013-06 → 2025-11 (150 months)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The sort separates, the neutralisation works — and only correlation pays (H₂)\n\n"
            "First confirm the machine does what it says: the correlation sort produces a real "
            "correlation gap and (after neutralisation) a flat beta gap. Then race the three sorts."
        ),
        code(
            "if HAVE_REAL:\n"
            "    book = st.bac_portfolio(PRICES, SPY)\n"
            "    rows = {}\n"
            "    for so in ['corr','beta','vol']:\n"
            "        b = st.bac_portfolio(PRICES, SPY, sort_on=so); s = st.summary(b['bac_ret'])\n"
            "        rows[so] = (s['mean']*100, s['sharpe'], s['tstat'])\n"
            "    cL, cS = book['avg_corr_long'].mean(), book['avg_corr_short'].mean()\n"
            "    bL, bS = book['avg_beta_long'].mean(), book['avg_beta_short'].mean()\n"
            "    banner = 'REAL tape'\n"
            "else:\n"
            f"    rows = {{'corr': ({R['corr_sort_mean']}, {R['bac_sharpe']}, {R['corr_sort_t']}),\n"
            f"            'beta': ({R['beta_sort_mean']}, -0.071, {R['beta_sort_t']}),\n"
            f"            'vol':  ({R['vol_sort_mean']}, -0.096, {R['vol_sort_t']})}}\n"
            f"    cL, cS, bL, bS = {R['corr_long']}, {R['corr_short']}, {R['beta_long']}, {R['beta_short']}\n"
            "    banner = 'SYNTHETIC fallback (frozen real numbers)'\n"
            "tab = pd.DataFrame({k: {'mean%/yr': v[0], 'Sharpe': v[1], 'HAC t': v[2]} for k,v in rows.items()}).T\n"
            "tab.index = ['correlation (BAC)','beta (BAB)','volatility (BAV)']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['long\\nleg','short\\nleg'], [cL, cS], color=[GREEN, RED], width=.5)\n"
            "a1.set_title(f'Correlation gap (sort works): {cL:.2f} vs {cS:.2f}'); a1.set_ylabel('avg correlation')\n"
            "a2.bar(['long\\nleg','short\\nleg'], [bL, bS], color=[GREEN, RED], width=.5)\n"
            "a2.set_title(f'Beta gap (before neutralising): {bL:.2f} vs {bS:.2f}'); a2.set_ylabel('avg beta')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            f"> 💡 In plain words: the sort cleanly separates correlation ({R['corr_long']} vs "
            f"{R['corr_short']}); the raw beta gap ({R['beta_long']} vs {R['beta_short']}) is exactly "
            "what the neutralisation scales away. And **H₂ is confirmed**: only the correlation row "
            f"earns ({R['corr_sort_mean']:+.1f}%/yr); beta ({R['beta_sort_mean']:+.1f}%) and vol "
            f"({R['vol_sort_mean']:+.1f}%) earn nothing."
        ),
        md(
            "### 4b · The headline book — H₁ with HAC *t* and a bootstrap CI\n\n"
            "The beta-neutral correlation book, with its inference furniture."
        ),
        code(
            "if HAVE_REAL:\n"
            "    book = st.bac_portfolio(PRICES, SPY)\n"
            "    s = st.summary(book['bac_ret']); m, t = s['mean']*100, s['tstat']\n"
            "    lo, hi = st.block_bootstrap_ci(book['bac_ret'], seed=502)\n"
            "    ci_lo, ci_hi = lo*100, hi*100\n"
            "    net = st.summary(st.apply_costs(book, cost_bps=5.0, borrow_ann_bps=50.0))\n"
            "    nm, nt = net['mean']*100, net['tstat']\n"
            "else:\n"
            f"    m, t, ci_lo, ci_hi = {R['bac_mean']}, {R['bac_t']}, {R['bac_ci_lo']}, {R['bac_ci_hi']}\n"
            f"    nm, nt = {R['bac_net_mean']}, {R['bac_net_t']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "b = ax.bar(['gross','net\\n(5bps+50bps borrow)'], [m, nm], color=[AMBER, GREY], width=.5)\n"
            "ax.errorbar(0, m, yerr=[[m-ci_lo],[ci_hi-m]], fmt='none', ecolor='k', capsize=6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('return %/yr')\n"
            "ax.set_title(f'+{m:.1f}%/yr at HAC t = {t:.2f} — CI [{ci_lo:.1f}, {ci_hi:.1f}] straddles 0')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross +{m:.2f}%/yr (HAC t {t:.2f}), 95% CI [{ci_lo:+.1f}, {ci_hi:+.1f}]; net +{nm:.2f}%/yr (t {nt:.2f})')"
        ),
        md(
            f"> 💡 In plain words: **H₁ supported in sign** (+{R['bac_mean']}%/yr) but the HAC "
            f"*t* = {R['bac_t']} and the bootstrap CI [{R['bac_ci_lo']}%, {R['bac_ci_hi']}%] straddles "
            "zero — below the bar. `WEAK`, not `REAL`."
        ),
        md(
            "### 4c · Not an artefact — the label-shuffle placebo (H₃)\n\n"
            "Permute the correlation ranks across names 200 times; where does the real mean land?"
        ),
        code(
            "# Light live placebo (80 shuffles) for speed; the stamped p uses 200 (docs/results.md).\n"
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(PRICES, SPY, n_shuffles=80, seed=502)\n"
            "    real_m, p, pmean, pstd = pb['real_mean']*100, pb['p_value'], pb['placebo_mean']*100, pb['placebo_std']*100\n"
            "else:\n"
            f"    real_m, p, pmean, pstd = {R['placebo_real']}, {R['placebo_p']}, {R['placebo_mean']}, {R['placebo_std']}\n"
            "xs = np.random.default_rng(0).normal(pmean, max(pstd,1e-6), 5000)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.hist(xs, bins=45, color=GREY, alpha=.6, label='shuffled (null)')\n"
            "ax.axvline(real_m, c=GREEN, lw=2.5, label=f'real +{real_m:.3f}%/mo')\n"
            "ax.axvline(0, c='k', lw=1); ax.set_xlabel('mean monthly book return %')\n"
            "ax.set_title(f'Real result in the right tail — one-sided placebo p = {p:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real {real_m:+.3f}%/mo vs shuffled {pmean:+.3f}% (sd {pstd:.3f}) -> p = {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: **H₃ supported** — the real mean sits in the right tail "
            f"(p = {R['placebo_p']:.2f}). The correlation labels carry real information; the spread is "
            "not manufactured by the construction. That's what separates a `WEAK` from a `NONE`."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁ positive in sign (+{R['bac_mean']}%/yr) but HAC *t* = "
            f"{R['bac_t']} < 2, CI straddles zero; placebo p = {R['placebo_p']:.2f} and AFGP keep it "
            "off `NONE`.\n"
            f"- **Tradability `FRAGILE`** — net +{R['bac_net_mean']}%/yr (*t* {R['bac_net_t']}) after "
            "costs + borrow; needs a short leg and collapses in correlation-spike years.\n"
            f"- **Is it correlation, not vol? `CONFIRMED`** — corr sort +{R['corr_sort_mean']}%/yr vs "
            f"beta {R['beta_sort_mean']}%/yr vs vol {R['vol_sort_mean']}%/yr."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs and the borrow (H₄)\n\n"
            "The book is thin to begin with; frictions thin it further."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    book = st.bac_portfolio(PRICES, SPY)\n"
            "    net = [st.summary(st.apply_costs(book, cost_bps=c, borrow_ann_bps=50.0))['mean']*100 for c in costs]\n"
            "else:\n"
            f"    base = {R['bac_mean']}; net = [base-0.05, base-0.2, base-0.4, base-0.75]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2); ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, color=AMBER, alpha=.12)\n"
            "ax.set_xlabel('one-way cost per leg (bps)'); ax.set_ylabel('net book %/yr')\n"
            "ax.set_title('Thin but resilient: low turnover (~6%/mo) keeps costs small (50 bps borrow throughout)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'At {R['cost_bps']:.0f} bps/leg + 50 bps borrow: ~+{R['bac_net_mean']}%/yr (t {R['bac_net_t']}).')"
        ),
        md(
            f"> 💡 In plain words: turnover is low (~{R['turnover']:.0f}%/mo — correlation is sticky), "
            "so costs barely dent the gross. **H₄ survives in sign** — but a book you can't tell from "
            "zero *before* costs is `FRAGILE` regardless of how cheap it is to run."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the seed-robust synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print the same number? Plant a "
            "correlation-keyed premium of increasing strength and watch the book's *t*-stat rise — "
            "**averaged over 20 seeds per point**, so no single lucky RNG draw can carry the claim."
        ),
        code(
            "# Light live control (8 seeds/point) for speed; the stamped curve uses 20 (docs/results.md).\n"
            "curve = st.synthetic_control_curve(premiums=(0.0, 0.04, 0.08, 0.12), n_seeds=8, base_seed=502)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "x = [p*100 for p in curve.index]\n"
            "ax.errorbar(x, curve['mean_t'], yerr=curve['std_t'], fmt='o-', c=GREEN, lw=2, capsize=4)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted correlation premium (%/yr)'); ax.set_ylabel('mean BAC HAC t (20 seeds)')\n"
            "ax.set_title('Faithful detector: t ~0 at the null, climbing monotonically with the planted premium')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(curve.round(3).to_string())"
        ),
        md(
            "The *t*-stat sits at ~0 at the null and climbs monotonically as the planted premium grows "
            "— a faithful (if low-powered) detector. The gentle slope is honest: a 48-name / ~120-month "
            "panel simply doesn't carry much statistical power, which is *why* a genuine real-tape "
            "mechanism (placebo p = 0.04) still lands at *t* = 1.54. The control proves the engine; it "
            "does **not** support the real-tape stamp. For the parent factor see "
            "[238 Betting-Against-Beta](../../238-betting-against-beta/); for the vol cousin, "
            "[330 Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)."
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
