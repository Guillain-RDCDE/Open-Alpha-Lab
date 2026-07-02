"""Generate the two narrative notebooks for Study 579 (Equity-Bond-Corr-Flip).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached SPY/TLT prices under
../_cache/ if present and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebooks re-run for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; SPY+TLT,
# 286 complete months 2002-08 -> 2026-05; price fp 884aad7ef4eb, monthly-panel fp 55f8c175e893).
R = dict(
    fp_px="884aad7ef4eb", fp_month="55f8c175e893",
    n_months=286, start="2002-08", end="2026-05",
    # regime split on forward 60/40 (%/mo):
    neg_mean=0.885, pos_mean=0.566, spread=0.319, split_t=0.89, placebo_p=0.36,
    n_neg=158, n_pos=123,
    # equity-leg split:
    eq_neg=1.089, eq_pos=0.899, eq_spread=0.19, eq_t=0.37, eq_p=0.72,
    # timer overlay (de-risk to 0.4x, 5 bps/switch):
    static_ann=8.75, timer_net_ann=6.90, edge_net_ann=-1.84,
    static_vol=9.9, timer_vol=7.1, static_sharpe=0.88, timer_sharpe=0.97,
    static_mdd=-28.5, timer_mdd=-23.6, n_switch=43,
    static_cum=610.3, timer_net_cum=385.8,
    y2022_6040=-14.0, pos_months_2022=12,
    # robustness windows: (label, spread %/mo, t, n_neg, n_pos)
    win=[("2002-2009", -0.300, -0.42, 55, 29), ("2010-2019", 0.613, 1.49, 87, 33),
         ("2020-2026", 0.636, 0.60, 16, 61), ("pre-2022 flip", 0.174, 0.47, 154, 74),
         ("2022-onward", 1.257, 1.20, 4, 49)],
    # synthetic control: (regime_edge, mean split-t over 25 seeds)
    ctrl=[(0.0, -0.12), (0.005, 1.34), (0.010, 2.80), (0.020, 5.73)],
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

from equity_bond_corr_flip import data, strategy as st

def load_real():
    \"\"\"Cache-first real monthly panel (empty frame offline).\"\"\"
    px = data.fetch_prices(fetch=False)
    if px.empty:
        return pd.DataFrame()
    return data.monthly_from_prices(px)

M = load_real()
HAVE_REAL = not M.empty
print("real SPY/TLT monthly panel present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(M)} months, {M.index.min().date()} -> {M.index.max().date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Stock-Bond Correlation Flip — does its sign tell you when 60/40 breaks? 🪢\n"
            "### When bonds stop hedging stocks, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n\n"
            "For twenty years, stocks and bonds moved *opposite* each other: when stocks fell, "
            "bonds rose, so a classic **60% stocks / 40% bonds** portfolio had a built-in shock "
            "absorber. Then **2022** happened — inflation surged, and stocks *and* bonds fell "
            "**together**. The 60/40 had its worst year in generations. The technical way to say it: "
            "the **stock-bond correlation flipped from negative to positive**.\n\n"
            "So here's the folk trading rule we test: *watch the sign of that correlation. When it "
            "goes positive, your bond hedge has stopped working — cut your 60/40 (de-risk).* Does "
            "the sign actually warn you in time, and does acting on it help?\n\n"
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
            f"| Does 60/40 do better after 'hedge working' (negative-corr) months? | **Yes, a little.** "
            f"**+{R['neg_mean']}%/mo** after negative-corr months vs **+{R['pos_mean']}%/mo** after "
            f"positive-corr ones — the right direction. |\n"
            f"| Is that gap real or noise? | **Can't tell it from noise.** The statistical test scores "
            f"*t* = {R['split_t']} (you need ~2), and a shuffle test gives *p* = {R['placebo_p']}. |\n"
            f"| Did the signal at least flag 2022? | **Yes.** The correlation went positive in early "
            f"2021 and stayed positive all through 2022 (a **{R['y2022_6040']}%** year for 60/40). |\n"
            f"| So should you time it? | **Only as a brake, not a gas pedal.** De-risking on the flip "
            f"cut the worst drawdown (**{R['static_mdd']}% → {R['timer_mdd']}%**) but *cost* "
            f"**{R['edge_net_ann']}%/yr** of return. |\n\n"
            "> The flip is real and the signal saw it — but 'de-risk on positive correlation' is a "
            "risk *reducer*, not a money maker, and the whole idea leans on a single big regime "
            "change in 24 years of data."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When the stock-bond correlation flips positive, the 60/40 hedge stops working — so "
            "cut your exposure.\"*\n\n"
            "Why it *should* matter: the entire appeal of adding bonds to a stock portfolio is "
            "diversification — bonds are supposed to zig when stocks zag. That only works when the "
            "two are **negatively** correlated. If they start falling together (positive "
            "correlation), the 40% in bonds no longer cushions anything, and your 'balanced' "
            "portfolio is just a slightly-less-volatile stock portfolio. 2022 was the poster child."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If the *sign* of a slow-moving correlation could warn you before the hedge fails, every "
            "risk-parity and 60/40 investor on Earth would want it. It's the cleanest possible "
            "market-timing story: one number, one sign, one decision. The desk has looked at the "
            "*broad* correlation regime ([578](../../578-cross-asset-correlation-regime/)) and at "
            "correlation as a stock-picking signal ([502](../../502-betting-against-correlation/)); "
            "here we go straight at **the one pair that matters for 60/40** — stocks vs bonds."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the correlation sign.** Each month, look at the last 6 months of SPY "
            "(stocks) and TLT (long bonds) returns and ask: are they moving together (positive) or "
            "opposite (negative)?\n"
            "2. **Look forward.** Split every month into 'negative-corr' and 'positive-corr', then "
            "compare how the 60/40 did *the next month*.\n"
            "3. **Test the rule.** Would de-risking whenever the sign is positive have helped — in "
            "return, in volatility, in the worst drawdown?\n\n"
            "Catch: the famous flip is essentially **one event** (2022). You can't learn much about "
            "'when regimes change' from a single change — a limitation we keep front and centre.\n\n"
            "*Timing:* the correlation sign this month uses only the past 6 months (fully known "
            "today); we act on it starting today and hold one month. No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Does 60/40 really do better after 'hedge-working' months?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.regime_split(M, 'fwd_6040'); ts = st.split_tstat(sp)\n"
            "    p = st.placebo_pvalue(M, 'fwd_6040', 2000, 579)\n"
            "    neg_m, pos_m, spr, t = sp['neg_mean']*100, sp['pos_mean']*100, sp['spread']*100, ts['t']\n"
            "    banner = f'REAL tape (SPY/TLT, {len(M)} months)'\n"
            "else:\n"
            f"    neg_m, pos_m, spr, t, p = {R['neg_mean']}, {R['pos_mean']}, {R['spread']}, {R['split_t']}, {R['placebo_p']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['after NEG-corr\\n(hedge working)','after POS-corr\\n(hedge broken)'], [neg_m, pos_m],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward 60/40 return %/mo')\n"
            "ax.set_title(f'60/40 does better after hedge-working months (+{neg_m:.2f} vs +{pos_m:.2f} %/mo) - but barely')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'neg +{neg_m:.3f}%/mo | pos +{pos_m:.3f}%/mo | spread {spr:+.3f}%/mo | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"So the direction is **right** — 60/40 earns more after the hedge-is-working months "
            f"(+{R['neg_mean']}%/mo vs +{R['pos_mean']}%/mo). But look at the *t*-stat: **{R['split_t']}**, "
            f"nowhere near the ~2 you'd want, and the shuffle test says *p* = {R['placebo_p']}. The gap "
            "is real in sign but drowned in noise."
        ),
        md(
            "**Is the sign at least *stable*?** Run the same split over different eras:"
        ),
        code(
            "wins = " + repr([(w[0], w[1]) for w in R["win"]]) + "\n"
            "if HAVE_REAL:\n"
            "    spans = [('2002-08-31','2009-12-31'),('2010-01-01','2019-12-31'),\n"
            "             ('2020-01-01','2026-05-31'),('2002-08-31','2021-12-31'),('2022-01-01','2026-05-31')]\n"
            "    labs = [w[0] for w in wins]; spr = []\n"
            "    for (sp0,en) in spans:\n"
            "        s = st.regime_split(M.loc[sp0:en], 'fwd_6040')\n"
            "        spr.append(s['spread']*100 if s else np.nan)\n"
            "else:\n"
            "    labs = [w[0] for w in wins]; spr = [w[1] for w in wins]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in spr]\n"
            "ax.bar(range(len(labs)), spr, color=cols, width=.6)\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=15)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('neg - pos spread %/mo')\n"
            "ax.set_title('The direction (green) mostly holds - but flips negative in 2002-09, none significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by window:', [round(s,3) for s in spr])"
        ),
        md(
            "The bars are *mostly* the folklore direction (green), but the earliest era (2002-09) is "
            "the **wrong** sign, and no window is statistically convincing. Worse, the biggest bar "
            "(2022-onward) rests on just **4** negative-correlation months — you can't measure a "
            "before/after contrast from 4 data points on one side."
        ),
        md(
            "**But did the signal at least catch 2022 in time — and did de-risking help?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = st.timer_overlay(M, cost_bps=5.0, derisk_to='bond')\n"
            "    s_ann, t_ann = tim['static_ann']*100, tim['timer_net_ann']*100\n"
            "    s_mdd, t_mdd = None, None\n"
            "    df = M.dropna(subset=['fwd_6040']); rr = df['fwd_6040'].to_numpy()\n"
            "    sign = df['corr_sign'].to_numpy(); stance = np.where(sign>0, 0.4, 1.0)\n"
            "    tr = stance*rr\n"
            "    def mdd(x):\n"
            "        c=np.cumprod(1+x); pk=np.maximum.accumulate(c); return float((c/pk-1).min())\n"
            "    s_mdd, t_mdd = mdd(rr)*100, mdd(tr)*100\n"
            "else:\n"
            f"    s_ann, t_ann, s_mdd, t_mdd = {R['static_ann']}, {R['timer_net_ann']}, {R['static_mdd']}, {R['timer_mdd']}\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))\n"
            "ax[0].bar(['static 60/40','corr-timer'], [s_ann, t_ann], color=[GREY, AMBER], width=.5)\n"
            "ax[0].set_ylabel('annualised return %'); ax[0].set_title('Return: timer gives some up')\n"
            "ax[1].bar(['static 60/40','corr-timer'], [s_mdd, t_mdd], color=[RED, GREEN], width=.5)\n"
            "ax[1].set_ylabel('max drawdown %'); ax[1].set_title('Drawdown: timer cushions it')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'return {s_ann:.2f}% -> {t_ann:.2f}%/yr | max drawdown {s_mdd:.1f}% -> {t_mdd:.1f}%')"
        ),
        md(
            f"There's the honest picture. De-risking on the positive-correlation signal **did** cushion "
            f"the crash — max drawdown improved **{R['static_mdd']}% → {R['timer_mdd']}%** — because the "
            f"signal really was positive through all of 2022's **{R['y2022_6040']}%** 60/40 year. But it "
            f"**cost return**: {R['static_ann']}% → {R['timer_net_ann']}%/yr, because the smaller book "
            "also sat out much of the 2023-25 recovery. A brake, not a gas pedal."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Right direction (+{R['neg_mean']}% vs +{R['pos_mean']}%/mo) but "
            f"*t* {R['split_t']} (placebo *p* {R['placebo_p']}), and the sign flips in the earliest "
            "era. The mechanism is real; the tape can't certify a bankable edge.\n"
            f"- **Tradability — Fragile.** A de-risking overlay that cuts drawdown "
            f"({R['static_mdd']}% → {R['timer_mdd']}%) and vol ({R['static_vol']}% → {R['timer_vol']}%) "
            f"but costs {R['edge_net_ann']}%/yr of return — a risk tool, not alpha, resting on one "
            "regime change."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "As a *risk* rule, sort of — de-risking when the correlation is positive genuinely "
            "trimmed the 2022 drawdown, and it nudged the risk-adjusted return (Sharpe) up a hair. "
            "As a *return* rule, no — you give up ~1.8%/yr for that comfort, and across a full cycle "
            "the return you sacrifice outweighs the drawdown you save. And remember: the entire case "
            "leans on **one** flip (2022). One correct call is not an edge you can compound."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The broad version.** [578 Cross-Asset-Correlation-Regime](../../578-cross-asset-correlation-regime/) "
            "does this for the *whole* correlation matrix, not just stocks vs bonds.\n"
            "- **The drivers.** [152 Inflation-Hedge](../../152-inflation-hedge/) and "
            "[119 Real-Rate-Regime](../../119-real-rate-regime/) are the macro forces *behind* the flip.\n\n"
            "*Think the flip is a tradable timer? Fork this, add more regime changes (1970s "
            "inflation, if you can source the data), and show the neg-minus-pos spread clearing "
            "t = 2 out of sample.*"
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
            "# The Stock-Bond Correlation Flip — a quantitative teardown 🔬\n"
            "### trailing-corr sign · regime split · two-sample *t* · label-shuffle placebo · five-window sign check · de-risk timer (return/vol/drawdown/Sharpe) · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We compute the sign of the "
            "trailing 6-month SPY/TLT return correlation and test whether it predicts the forward "
            "60/40 return — and whether timing on it pays.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily SPY + TLT, "
            f"{R['n_months']} complete months {R['start']} → {R['end']} (as-of 2026-06-30, monthly-panel "
            f"fp `{R['fp_month']}`); the offline core and tests run on a deterministic synthetic world. "
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
            f"| **Signal** | `WEAK` | Forward 60/40 neg − pos spread **+{R['spread']}%/mo** (two-sample "
            f"*t* **+{R['split_t']}**, placebo *p* {R['placebo_p']}) — right sign, sub-2 *t*; sign flips "
            f"across sub-windows (−0.30 in 2002-09). |\n"
            f"| **Tradability** | `FRAGILE` | De-risk timer: max DD **{R['static_mdd']}% → {R['timer_mdd']}%**, "
            f"vol **{R['static_vol']}% → {R['timer_vol']}%**, Sharpe **{R['static_sharpe']} → {R['timer_sharpe']}** "
            f"— but net return edge **{R['edge_net_ann']}%/yr**. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on the real "
            "SPY/TLT tape the regime split is directionally right yet sub-threshold, and its tradable "
            "form only reduces risk while costing return."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_t \\in \\{-1,+1\\}$ be the sign of the trailing 6-month $\\text{corr}(r^{eq}, r^{bd})$ "
            "at month $t$, and $r^{60/40}_{t+1}$ the forward one-month 60/40 return.\n\n"
            "- **H₁ (regime split).** $\\mathbb{E}[r^{60/40}_{t+1} \\mid s_t<0] - "
            "\\mathbb{E}[r^{60/40}_{t+1} \\mid s_t>0] > 0$ at *t* ≥ 2 (60/40 does better when the hedge "
            "works).\n"
            "- **H₂ (robustness).** The sign of H₁ is stable across sub-periods.\n"
            "- **H₃ (timing).** A de-risk-when-$s_t>0$ overlay beats static 60/40 net of costs.\n\n"
            "We find **H₁ directionally right but not significant** (*t* = +0.89), **H₂ rejected** "
            "(sign flips, none significant), and **H₃ split**: the timer *cuts risk* but *loses "
            "return*."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The stock-bond correlation regime is well grounded "
            "(Campbell-Sunderam-Viceira 2017: growth-shock regimes → negative corr / bonds hedge; "
            "inflation-shock regimes → positive corr / bonds and stocks fall together). 2022 was a "
            "textbook inflation-regime flip. Two things stop it from being a *timer*: (a) it is "
            "**one** regime change in 24 years — a tiny effective sample for a before/after contrast; "
            "and (b) the tradable rule is a **de-risking** overlay, which structurally trades return "
            "for lower risk in a market that mostly rises. That maps to the broad-correlation reading "
            "in [578](../../578-cross-asset-correlation-regime/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $s_t = \\text{sign}\\,\\text{corr}_{6m}(r^{eq}, r^{bd})$, from the trailing "
            "6 monthly returns (public at month $t$'s close).\n"
            "- **Split.** Group forward one-month 60/40 (and equity) returns by $s_t$; two-sample "
            "(Welch) *t* on neg − pos.\n"
            "- **Placebo.** Shuffle the $s_t$ labels vs forward returns (2000 perms) — the null.\n"
            "- **Robustness.** Re-split over five date windows; read sign and *t*.\n"
            "- **Timing.** Hold static 60/40 when $s_t<0$, de-risk to 0.4× when $s_t>0$; 5 bps/switch, "
            "long-only (no borrow). Compare return / vol / drawdown / Sharpe, gross and net.\n"
            "- **Positive control.** A deterministic synthetic world with a planted regime penalty "
            "(`regime_edge > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: $s_t$ uses only trailing data, so it is public at month $t$'s close; the timer "
            "enters at that close and holds one month. One documented execution lag; a one-month-later "
            "entry is a minor perturbation (the hold is a full month)."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The regime split — H₁\n\n"
            "Forward 60/40 returns by trailing-corr sign, with the two-sample *t* and the placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.regime_split(M, 'fwd_6040'); ts = st.split_tstat(sp)\n"
            "    p = st.placebo_pvalue(M, 'fwd_6040', 2000, 579)\n"
            "    neg_m, pos_m, spr, t = sp['neg_mean']*100, sp['pos_mean']*100, sp['spread']*100, ts['t']\n"
            "    nn, npos = sp['n_neg'], sp['n_pos']\n"
            "else:\n"
            f"    neg_m, pos_m, spr, t, p = {R['neg_mean']}, {R['pos_mean']}, {R['spread']}, {R['split_t']}, {R['placebo_p']}\n"
            f"    nn, npos = {R['n_neg']}, {R['n_pos']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['neg-corr regime','pos-corr regime'], [neg_m, pos_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward 60/40 %/mo')\n"
            "ax.set_title(f'neg - pos = {spr:+.3f}%/mo  (two-sample t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'neg +{neg_m:.3f}%/mo (n={nn}) | pos +{pos_m:.3f}%/mo (n={npos}) | spread {spr:+.3f}%/mo | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ is **directionally right but not significant.** The spread is "
            f"**+{R['spread']}%/mo** in the folklore's direction, but at *t* {R['split_t']} (placebo *p* "
            f"{R['placebo_p']}) you cannot reject 'the sign tells you nothing.' The equity-only leg is "
            f"weaker still (spread +{R['eq_spread']}%/mo, *t* +{R['eq_t']})."
        ),
        md(
            "### 4b · Robustness — H₂ (does the sign hold?)\n\n"
            "The same split over five date windows — read the sign and the effective sample on each "
            "side."
        ),
        code(
            "wins = " + repr(R["win"]) + "\n"
            "if HAVE_REAL:\n"
            "    spans = [('2002-08-31','2009-12-31'),('2010-01-01','2019-12-31'),\n"
            "             ('2020-01-01','2026-05-31'),('2002-08-31','2021-12-31'),('2022-01-01','2026-05-31')]\n"
            "    labs = [w[0] for w in wins]\n"
            "    rows = st.window_sweep(M, list(zip(labs, [s[0] for s in spans], [s[1] for s in spans])), 'fwd_6040')\n"
            "    tab = pd.DataFrame([(l, s*100 if s==s else np.nan, t, nn, np_) for (l,s,t,nn,np_) in rows],\n"
            "                       columns=['window','spread%/mo','t','n_neg','n_pos']).set_index('window')\n"
            "else:\n"
            "    tab = pd.DataFrame([(w[0], w[1], w[2], w[3], w[4]) for w in wins],\n"
            "                       columns=['window','spread%/mo','t','n_neg','n_pos']).set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in tab['spread%/mo']]\n"
            "ax.bar(range(len(tab)), tab['spread%/mo'], color=cols, width=.6)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index, rotation=15)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('neg - pos spread %/mo')\n"
            "ax.set_title('Sign mostly folklore-direction, wrong in 2002-09, none significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₂ **rejected.** The spread is *negative* in 2002-09 and positive "
            "afterward, and **no window clears *t* = 2**. The 2022-onward window has the biggest spread "
            "(+1.26%/mo) but only **4** negative-corr months against 49 positive — you cannot estimate "
            "a contrast from 4 observations of one side."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁ directionally right (spread +{R['spread']}%/mo) but *t* "
            f"+{R['split_t']} (placebo *p* {R['placebo_p']}); H₂ rejected (sign flips, none "
            "significant).\n"
            f"- **Tradability `FRAGILE`** — the timer cuts drawdown ({R['static_mdd']}% → "
            f"{R['timer_mdd']}%) and vol ({R['static_vol']}% → {R['timer_vol']}%) but costs "
            f"{R['edge_net_ann']}%/yr; a risk overlay, not alpha."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the de-risk timer\n\n"
            "Hold static 60/40 when the correlation is negative, de-risk to 0.4× when positive, "
            "5 bps/switch. Look at return, vol, drawdown and Sharpe together."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = st.timer_overlay(M, cost_bps=5.0, derisk_to='bond')\n"
            "    s_ann, t_ann = tim['static_ann']*100, tim['timer_net_ann']*100\n"
            "    s_vol, t_vol = tim['static_vol']*100, tim['timer_vol']*100\n"
            "    s_sh, t_sh = tim['static_sharpe'], tim['timer_sharpe']\n"
            "    df = M.dropna(subset=['fwd_6040']); rr = df['fwd_6040'].to_numpy()\n"
            "    sign = df['corr_sign'].to_numpy(); tr = np.where(sign>0,0.4,1.0)*rr\n"
            "    def mdd(x):\n"
            "        c=np.cumprod(1+x); pk=np.maximum.accumulate(c); return float((c/pk-1).min())\n"
            "    s_mdd, t_mdd = mdd(rr)*100, mdd(tr)*100\n"
            "    edge = tim['edge_net_ann']*100\n"
            "else:\n"
            f"    s_ann, t_ann = {R['static_ann']}, {R['timer_net_ann']}\n"
            f"    s_vol, t_vol = {R['static_vol']}, {R['timer_vol']}\n"
            f"    s_sh, t_sh = {R['static_sharpe']}, {R['timer_sharpe']}\n"
            f"    s_mdd, t_mdd, edge = {R['static_mdd']}, {R['timer_mdd']}, {R['edge_net_ann']}\n"
            "fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))\n"
            "ax[0].bar(['static','timer'], [s_ann, t_ann], color=[GREY, AMBER], width=.5); ax[0].set_title('ann return %'); ax[0].axhline(0,c='k',lw=1)\n"
            "ax[1].bar(['static','timer'], [s_vol, t_vol], color=[GREY, AMBER], width=.5); ax[1].set_title('ann vol %')\n"
            "ax[2].bar(['static','timer'], [s_mdd, t_mdd], color=[RED, GREEN], width=.5); ax[2].set_title('max drawdown %')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'return {s_ann:.2f}% -> {t_ann:.2f}%/yr (edge {edge:+.2f}%/yr) | vol {s_vol:.1f}% -> {t_vol:.1f}% | Sharpe {s_sh:.2f} -> {t_sh:.2f} | maxDD {s_mdd:.1f}% -> {t_mdd:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: the timer is a **risk reducer, not an alpha source.** It trims vol "
            f"({R['static_vol']}% → {R['timer_vol']}%) and drawdown ({R['static_mdd']}% → "
            f"{R['timer_mdd']}%) and nudges Sharpe ({R['static_sharpe']} → {R['timer_sharpe']}), but "
            f"gives up **{R['edge_net_ann']}%/yr** of return. `FRAGILE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real regime effect "
            "(forward 60/40 worse after positive-corr months, `regime_edge > 0`) of increasing "
            "strength and watch the split-*t* climb — averaged over 25 seeds so no single lucky seed "
            "can fake it."
        ),
        code(
            "edges = [0.0, 0.003, 0.005, 0.010, 0.015, 0.020]\n"
            "ts = [st.synthetic_mean_t(data, regime_edge=e, n_seeds=25) for e in edges]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(edges, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted regime_edge (stronger = bigger hedge-break penalty)')\n"
            "ax.set_ylabel('mean split-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted regime effect - flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e, t in zip(edges, ts): print(f'regime_edge {e:+.3f} -> mean split-t {t:+.2f}')"
        ),
        md(
            "The split-*t* is ≈ 0 at the null and rises past +2 as the planted effect grows — so the "
            "engine is a faithful detector. The sub-2 real-tape *t* is therefore a statement about "
            "**the data**: the correlation-sign timer is directionally right but not significant, and "
            "its tradable form only reduces risk. For the broad-correlation cousin see "
            "[578 Cross-Asset-Correlation-Regime](../../578-cross-asset-correlation-regime/)."
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
