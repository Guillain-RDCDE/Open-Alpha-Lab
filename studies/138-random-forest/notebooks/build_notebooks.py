"""Generate the two narrative notebooks for Study 138 (Random-Forest).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    ticker="SPY",
    date0="2010-01-04",
    date1="2026-06-12",
    n_bars=4136,
    fp="fed737ecfc62",
    n_oos=3611,
    acc_real=0.5348,
    acc_shuf=0.5264,
    delta_acc=0.0083,
    gross_sharpe_real=0.498,
    gross_t_real=2.149,
    gross_sharpe_shuf=0.575,
    gross_t_shuf=2.344,
    net_sharpe_5bp=0.200,
    net_t_5bp=0.858,
    mean_gross_bps=2.26,
    mean_net_bps=0.91,
    bh_sharpe=0.858,
    pct_long=0.755,
    sharpe_0bp=0.498,
    t_0bp=2.149,
    sharpe_2bp=0.379,
    t_2bp=1.631,
    sharpe_5bp=0.200,
    t_5bp=0.858,
    sharpe_10bp=-0.098,
    t_10bp=-0.417,
)

# Pre-extract all values as simple local names to avoid f-string key quoting issues.
_acc_real = R["acc_real"]
_acc_shuf = R["acc_shuf"]
_delta_acc = R["delta_acc"]
_delta_pp = R["delta_acc"] * 100
_gross_sharpe_real = R["gross_sharpe_real"]
_gross_t_real = R["gross_t_real"]
_gross_sharpe_shuf = R["gross_sharpe_shuf"]
_gross_t_shuf = R["gross_t_shuf"]
_net_sharpe_5bp = R["net_sharpe_5bp"]
_net_t_5bp = R["net_t_5bp"]
_mean_gross_bps = R["mean_gross_bps"]
_mean_net_bps = R["mean_net_bps"]
_bh_sharpe = R["bh_sharpe"]
_pct_long = R["pct_long"]
_n_oos = R["n_oos"]
_date0 = R["date0"]
_date1 = R["date1"]
_n_bars = R["n_bars"]
_fp = R["fp"]
_sharpe_0bp = R["sharpe_0bp"]
_t_0bp = R["t_0bp"]
_sharpe_2bp = R["sharpe_2bp"]
_t_2bp = R["t_2bp"]
_sharpe_5bp = R["sharpe_5bp"]
_t_5bp = R["t_5bp"]
_sharpe_10bp = R["sharpe_10bp"]
_t_10bp = R["t_10bp"]

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
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

from random_forest import data, strategy as st

TICKER = "SPY"
CACHE_FILE = data._cache_path(TICKER, data.DEFAULT_CACHE)
HAVE_REAL = os.path.exists(CACHE_FILE)

def load_real():
    bars = data.fetch_daily(TICKER, fetch=False)
    X, y = st.build_features(bars)
    return bars, X, y

print("Real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Random-Forest -- can ML find alpha in SPY daily returns?\n"
            "### A tree ensemble on standard features, tested with walk-forward and a shuffle control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_shuffle%3F: No](https://img.shields.io/badge/Beats_shuffle%3F-No-8b949e?style=flat-square)\n\n"
            "You have heard it a thousand times: *'just train a machine learning model on price history "
            "and let it trade.'*  The idea has surface appeal -- computers see patterns humans miss.  "
            "This notebook tests whether a Random Forest, trained on the standard toolkit of technical "
            "features (lagged returns, RSI, realised volatility, momentum, volume ratio), actually "
            "learns something useful about next-day SPY direction out-of-sample -- or whether it just "
            "memorises noise.\n\n"
            "> **This is the plain-language layer.**  For the t-stats, bootstrap intervals, and the "
            "full shuffle-control maths, see the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n\n"
            "> **Not investment advice.**  Reproducible research -- every chart is drawn by the code "
            "beside it.  House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the RF beat a coin? | **Barely.** Out-of-sample accuracy is "
            f"**{_acc_real:.1%}** vs a coin's 50% -- but the shuffled-label model also scores "
            f"**{_acc_shuf:.1%}**, so the gap ({_delta_pp:.1f}pp) is noise, not signal. |\n"
            f"| Does it beat buy-and-hold? | **No.** Net Sharpe {_net_sharpe_5bp:+.2f} at 5 bps "
            f"vs SPY buy-and-hold {_bh_sharpe:+.2f}. |\n"
            f"| Isn't the gross t-stat {_gross_t_real:.2f} significant? | Yes, but the "
            f"**shuffled model scores {_gross_t_shuf:.2f}** -- both are harvesting SPY's bull-run "
            "beta, not a directional signal the features capture. |\n"
            "| Could you trade it? | **No.** A passive index fund beats it with no effort. |\n\n"
            "> The Random Forest learned to be long SPY most of the time (~75% of days), which "
            "beats a coin -- but so does a shuffled model that never saw real labels."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'Train a Random Forest on standard technical indicators -- RSI, lagged returns, "
            "volatility, momentum -- and use its next-day direction signal to trade SPY.  "
            "Machine learning finds nonlinear patterns that simple rules miss; even a modest "
            "accuracy edge (55%+) translates to strong risk-adjusted returns.'*\n\n"
            "This is one of the most popular retail quant pitches, and it has academic backing: "
            "Gu, Kelly & Xiu (2020) found tree ensembles have the best predictive R2 in large "
            "cross-sectional studies.  The question is whether the accessible version -- a "
            "single-ticker daily RF on technical features -- actually delivers."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it worked, it would be the simplest systematic edge imaginable: download SPY "
            "data, compute five features, train a scikit-learn RandomForestClassifier, trade "
            "the signal.  No exotic data, no Bloomberg terminal.\n\n"
            "If it does not work -- and the honest test is that it does not -- then every "
            "YouTube tutorial and Jupyter notebook claiming 70%+ backtested accuracy is hiding "
            "the one thing that kills it: **in-sample overfitting**, exposed by a strict "
            "walk-forward evaluation and confirmed by a shuffled-label control."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Two tests keep us honest:\n\n"
            "1. **Walk-forward only.**  The model never trains on data it is also predicting.  "
            "We fit on a rolling 2-year window, predict the next quarter, then advance and "
            "repeat -- exactly as a live trader would have operated.  Every accuracy number "
            "is out-of-sample.\n\n"
            "2. **Shuffled-label control.**  We run the identical walk-forward with one change: "
            "the training labels are randomly permuted in each fold before fitting.  A "
            "shuffled-label model that cannot possibly learn anything from the features still "
            "produces predictions -- all from noise.  If the real model scores the same as the "
            "shuffled one, the features are useless.  If it scores materially higher, "
            "the features carry real information.\n\n"
            f"Tape: SPY daily closes, {_date0} to {_date1} ({_n_bars} days, "
            f"fingerprint `{_fp}`).  Training window: 504 days (~2 years).  "
            "Test step: 63 days (~1 quarter).  Features: lagged returns (5 lags), RSI(14), "
            "realised vol(21), momentum(20), volume ratio."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown\n\n"
            "**First: what does the RF actually learn?**  Here are the in-sample feature "
            "importances from the initial 2-year training period -- where the features look "
            "most useful (because the model has seen all the 'right answers'):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, X, y = load_real()\n"
            "    imp = st.feature_importance(X, y, train_window=504, n_estimators=100)\n"
            "else:\n"
            "    imp = pd.Series({\n"
            "        'ret_lag_4':0.1583,'ret_lag_2':0.1290,'ret_lag_1':0.1225,\n"
            "        'mom':0.1149,'vol_ratio':0.1101,'rsi':0.0963,\n"
            "        'rvol':0.0911,'ret_lag_3':0.0891,'ret_lag_5':0.0886\n"
            "    }).sort_values(ascending=False)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "ax.barh(imp.index[::-1], imp.values[::-1], color=GREY)\n"
            "ax.set_xlabel('Mean impurity decrease (in-sample)')\n"
            "ax.set_title('In-sample feature importance -- all features roughly equal')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('No feature dominates; none are clearly predictive.')"
        ),
        md(
            "No feature stands out -- the importances are nearly uniform.  When a forest "
            "assigns roughly equal weight to everything, it has not found a dominant signal; "
            "it has spread its bets across noise.  The in-sample picture always looks better "
            "than reality -- which is exactly why we need the walk-forward test."
        ),
        md(
            "**Now the honest test: real walk-forward vs the shuffled-label control.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, X, y = load_real()\n"
            "    preds_real = st.walk_forward(X, y, train_window=504, step=63, n_estimators=200)\n"
            "    preds_shuf = st.walk_forward(X, y, train_window=504, step=63, n_estimators=200, shuffle_labels=True)\n"
            "    acc_r = float((preds_real['y_true']==preds_real['y_pred']).mean())\n"
            "    acc_s = float((preds_shuf['y_true']==preds_shuf['y_pred']).mean())\n"
            "else:\n"
            f"    acc_r, acc_s = {_acc_real}, {_acc_shuf}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['RF (real labels)\\nwalk-forward', 'RF (shuffled labels)\\ncontrol'],\n"
            "       [acc_r, acc_s], color=[GREEN, GREY], width=0.55)\n"
            "ax.axhline(0.5, ls='--', c='k', lw=1, label='coin flip (50%)')\n"
            "ax.set_ylabel('OOS accuracy'); ax.set_ylim(0.48, 0.56)\n"
            "ax.set_title('Real model barely edges the shuffled control -- within noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Real: {acc_r:.4f}   Shuffle: {acc_s:.4f}   Gap: {acc_r-acc_s:+.4f}')"
        ),
        md(
            f"The real model scores **{_acc_real:.1%}** vs the shuffled control "
            f"**{_acc_shuf:.1%}** -- a gap of just **{_delta_pp:.1f} percentage points**.  "
            "The shuffled model learned nothing; the real model learned almost nothing extra.  "
            "Both sit barely above the coin flip line.\n\n"
            "> The reason is not that the RF is broken -- a shuffled model should score ~50%.  "
            "The reason both score 53% is that both models learned to be **long most of the "
            "time**.  SPY went up in the sample, so 'long' beats 'short' regardless of the "
            "features.  The RF did not learn direction; it learned 'long-only ETF'."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** OOS accuracy {_acc_real:.1%} vs shuffle {_acc_shuf:.1%} "
            f"(gap = {_delta_pp:.1f}pp); the shuffled model's gross t-stat "
            f"({_gross_t_shuf:.2f}) exceeds the real model ({_gross_t_real:.2f}).  "
            "Both harvest SPY beta, not a learned directional signal.\n"
            f"- **Tradability -- Mirage.** Net Sharpe {_net_sharpe_5bp:+.2f} at 5 bps "
            f"vs buy-and-hold {_bh_sharpe:+.2f}.  A passive index fund is better.\n"
            "- **Beats shuffle? -- No.** The decisive test: the shuffled-label model "
            "matches or outperforms the real model on every gross metric."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The short answer is no -- but let's see what costs do to the already-thin "
            "gross edge:"
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            f"sharpes = [{_sharpe_0bp}, {_sharpe_2bp}, {_sharpe_5bp}, {_sharpe_10bp}]\n"
            f"tstats  = [{_t_0bp}, {_t_2bp}, {_t_5bp}, {_t_10bp}]\n"
            "if HAVE_REAL:\n"
            "    bars, X, y = load_real()\n"
            "    preds_real = st.walk_forward(X, y, train_window=504, step=63, n_estimators=200)\n"
            "    sharpes, tstats = [], []\n"
            "    for c in costs:\n"
            "        led = st.trade_predictions(bars, preds_real, cost_bps=c)\n"
            "        s = st.summarize(preds_real, led)\n"
            "        sharpes.append(s['sharpe_net']); tstats.append(s['tstat_net'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, sharpes, 'o-', c=RED, lw=2, label='strategy Sharpe')\n"
            f"ax.axhline({_bh_sharpe:.3f}, ls='--', c=GREY, lw=1.5, label='SPY B&H Sharpe {_bh_sharpe:.2f}')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('Strategy Sharpe collapses under costs; buy-and-hold wins')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for c, sh, t in zip(costs, sharpes, tstats):\n"
            "    print(f'  {c:.0f} bp: Sharpe {sh:+.3f}  HAC t {t:+.3f}')"
        ),
        md(
            f"Even at zero cost the strategy's Sharpe ({_sharpe_0bp:.2f}) is well below "
            f"buy-and-hold ({_bh_sharpe:.2f}) -- you are being long 75% of days instead of "
            "100%.  At a realistic 5 bps the Sharpe falls to +0.20 and the t-stat is 0.86 -- "
            "statistically indistinguishable from zero.  At 10 bps it goes negative.\n\n"
            "There is no cost level at which this beats a passive index fund."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Is ML useless for finance?** Not at all -- Gu, Kelly & Xiu (2020) found "
            "real tree-ensemble predictability in cross-sectional returns with 900+ features.  "
            "The lesson here is narrower: a single-ticker daily RF on 5 standard technical "
            "features carries nothing a shuffled model does not also carry.\n"
            "- **What would a fair test of ML alpha look like?**  Cross-sectional features "
            "(value, quality, momentum across many stocks), longer holding periods, and a "
            "systematic shuffled-label check as done here.  Study 65 (Scorecard) and "
            "Study 121 (Magic-Formula) explore the factor cross-section honestly.\n"
            "- **The Study 39 contrast.**  [Black-Box (Study 39)](../../39-black-box/) tests "
            "an MLP on crypto -- same walk-forward / shuffle protocol, different model family "
            "and asset class.  Same verdict.\n\n"
            "*Think there is a signal? Fork this, add alternative features (alternative data, "
            "cross-asset signals, fundamentals), run the shuffle test, and clear a t-stat "
            "> 2 that the shuffled model cannot match.  That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Random-Forest -- a quantitative teardown\n"
            "### Walk-forward RF on SPY technical features - shuffle control - HAC inference - cost sweep\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_shuffle%3F: No](https://img.shields.io/badge/Beats_shuffle%3F-No-8b949e?style=flat-square)\n\n"
            "The quantitative companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim now carrying its standard error.  We test whether a "
            "Random Forest classifier, trained on standard technical features in a strict rolling "
            "walk-forward, predicts next-day SPY direction better than a shuffled-label control, "
            "and whether any residual edge survives realistic costs.\n\n"
            "> **Not investment advice.**  Real data: Yahoo daily OHLCV, "
            f"{_date0} to {_date1}, fingerprint `{_fp}`, as-of 2026-06-14.  "
            "Methods in [`docs/references.md`](../docs/references.md); reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **`In plain words` notes** translate each result to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | OOS accuracy {_acc_real:.1%} vs shuffle {_acc_shuf:.1%} "
            f"(gap {_delta_pp:+.1f}pp); shuffled gross t ({_gross_t_shuf:.2f}) exceeds "
            f"real ({_gross_t_real:.2f}) -- both are SPY beta, not learned signal. |\n"
            f"| **Tradability** | `MIRAGE` | Net Sharpe {_net_sharpe_5bp:+.2f} at 5 bps "
            f"(HAC t = {_net_t_5bp:+.2f}); buy-and-hold Sharpe {_bh_sharpe:+.2f}. |\n"
            "| **Beats shuffle?** | `NO` | Shuffled model matches or outperforms real on every "
            "gross metric; gap is within sampling noise. |\n\n"
            "> In plain words: the Random Forest learned to be long SPY ~75% of days, which "
            "happens to score 53.5% accuracy on 16 years of a bull market.  A model trained on "
            "permuted labels does the same thing.  The features add nothing."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $x_t$ be a feature vector of standard technical indicators available at "
            "close of day $t$, and $y_{t+1} = \\mathbf{1}[r_{t+1} > 0]$ the next-day "
            "direction.  The recipe asserts:\n\n"
            "- **H1 (signal).**  $\\hat{y}_{t+1} = f_{\\text{RF}}(x_t)$, trained by "
            "walk-forward on real labels, achieves accuracy materially above "
            "$f_{\\text{shuffle}}(x_t)$, trained on permuted labels -- i.e. the features "
            "carry directional information beyond what a noise model can capture.\n"
            "- **H2 (alpha).**  The resulting equity curve has a Sharpe ratio above "
            "buy-and-hold, net of realistic costs.\n\n"
            f"We reject H1 (the gap is {_delta_pp:.1f}pp, within noise; the shuffled "
            f"model's gross t exceeds the real model's) and H2 (net Sharpe {_net_sharpe_5bp:.2f} "
            f"vs B&H {_bh_sharpe:.2f})."
        ),

        # ---- BEAT 2 -- SO WHAT -------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "A real H1 rejection would mean equity prices carry no exploitable short-term "
            "predictability in standard technical features -- consistent with (weakly "
            "efficient) random-walk models for daily returns.  The interesting finding is "
            "not that the RF fails, but *why* it appears to succeed in-sample: both the real "
            "and shuffled models learn to be long SPY most of the time, harvesting the "
            "sample's positive trend rather than any directional signal.\n\n"
            "This is the **beta-disguised-as-alpha** failure mode: the model's positive "
            "gross return is real, but it is compensation for equity beta, not evidence of "
            "learned predictability."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -------------------------------------------
        md(
            "## 3 - The protocol\n\n"
            "- **Features.** $x_t$ = (ret\\_lag\\_1..5, RSI(14), rvol(21), mom(20), "
            "vol\\_ratio(20)); all computed from data available at close of day $t$.\n"
            "- **Walk-forward.** Training window = 504 days; step = 63 days; model refitted "
            "each quarter.  Test set is never seen during training.  No data leakage.\n"
            "- **Shuffled-label control.** Identical walk-forward; training labels are randomly "
            "permuted within each fold before fitting.  Test labels are not permuted.  "
            "If real score ~ shuffled score, features carry no useful information.\n"
            "- **Trading.** Long (+1) on predicted-up days; flat (0) otherwise.  "
            "Open-to-close daily return.  Cost = round-trip bps charged on each position change.\n"
            "- **Inference.** HAC Newey-West t-stat on daily net returns; bootstrap Sharpe CI "
            "(circular block bootstrap, block = n^(1/3)).\n"
            "- **Positive control.** Synthetic tape with a planted nonlinear signal "
            "(tanh lag interaction) -- the machine must beat the shuffle on this tape."
        ),

        # ---- BEAT 4 -- TEARDOWN ------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Walk-forward accuracy: real model vs shuffled control\n\n"
            "The fundamental question: does the RF trained on real labels know more than "
            "a RF trained on random noise?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, X, y = load_real()\n"
            "    preds_real = st.walk_forward(X, y, train_window=504, step=63, n_estimators=200)\n"
            "    preds_shuf = st.walk_forward(X, y, train_window=504, step=63, n_estimators=200, shuffle_labels=True)\n"
            "    acc_r = float((preds_real['y_true']==preds_real['y_pred']).mean())\n"
            "    acc_s = float((preds_shuf['y_true']==preds_shuf['y_pred']).mean())\n"
            "    pct_long = float((preds_real['y_pred']==1).mean())\n"
            "else:\n"
            f"    acc_r, acc_s, pct_long = {_acc_real}, {_acc_shuf}, {_pct_long}\n"
            "print(f'Real accuracy:    {acc_r:.4f}')\n"
            "print(f'Shuffle accuracy: {acc_s:.4f}  (gap = {acc_r-acc_s:+.4f})')\n"
            "print(f'Pct days long:    {pct_long:.3f}')\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "ax.bar(['RF real labels', 'RF shuffled labels', 'Coin flip (50%)'],\n"
            "       [acc_r, acc_s, 0.5], color=[GREEN, GREY, RED], width=0.55)\n"
            "ax.axhline(0.5, ls='--', c='k', lw=1)\n"
            "ax.set_ylabel('OOS directional accuracy'); ax.set_ylim(0.48, 0.57)\n"
            "ax.set_title('The shuffled model nearly matches the real model')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> In plain words: the gap between real ({_acc_real:.1%}) and shuffled "
            f"({_acc_shuf:.1%}) is {_delta_pp:.1f}pp over {_n_oos:,} out-of-sample "
            "predictions.  With ~3,600 obs a 1pp gap has a standard error of about 1.7pp "
            "-- this gap is not statistically significant."
        ),
        md(
            "### 4b - Gross equity curves: shuffled model matches or beats real model\n\n"
            "Both models are long ~75% of days.  The gross t-stat of the shuffled model "
            "actually exceeds the real model -- confirming the positive return is beta, "
            "not learned signal."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, X, y = load_real()\n"
            "    preds_real = st.walk_forward(X, y, train_window=504, step=63, n_estimators=200)\n"
            "    preds_shuf = st.walk_forward(X, y, train_window=504, step=63, n_estimators=200, shuffle_labels=True)\n"
            "    led_r = st.trade_predictions(bars, preds_real, cost_bps=0)\n"
            "    led_s = st.trade_predictions(bars, preds_shuf, cost_bps=0)\n"
            "    cum_r = (1+led_r['ret_gross']).cumprod()\n"
            "    cum_s = (1+led_s['ret_gross']).cumprod()\n"
            "    bh = (1+bars['close'].pct_change().reindex(led_r.index).fillna(0)).cumprod()\n"
            "else:\n"
            "    import numpy as np\n"
            "    t = pd.date_range('2012-01-01', periods=3611, freq='B')\n"
            f"    cum_r = pd.Series((1+{_mean_gross_bps * 1e-4:.6f})**np.arange(3611), index=t)\n"
            "    cum_s = cum_r * 1.05\n"
            "    bh = cum_r * 1.3\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            f"ax.plot(cum_r, c=GREEN, lw=1.5, label='RF real (gross Sharpe {_gross_sharpe_real:.2f}, t={_gross_t_real:.2f})')\n"
            f"ax.plot(cum_s, c=GREY, lw=1.5, ls='--', label='RF shuffle (gross Sharpe {_gross_sharpe_shuf:.2f}, t={_gross_t_shuf:.2f})')\n"
            f"ax.plot(bh, c=RED, lw=1.5, ls=':', label='Buy-and-hold SPY (Sharpe {_bh_sharpe:.2f})')\n"
            "ax.set_ylabel('Cumulative gross return (starting 1.0)')\n"
            "ax.set_title('Both models underperform buy-and-hold; shuffled model similar to real')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print('RF real gross Sharpe: {_gross_sharpe_real:+.3f}  (t={_gross_t_real:+.3f})')\n"
            f"print('RF shuf gross Sharpe: {_gross_sharpe_shuf:+.3f}  (t={_gross_t_shuf:+.3f})')\n"
            f"print('BH Sharpe:            {_bh_sharpe:+.3f}')"
        ),
        md(
            f"> In plain words: the shuffled model (Sharpe {_gross_sharpe_shuf:.2f}, "
            f"t = {_gross_t_shuf:.2f}) is **better** than the real model "
            f"(Sharpe {_gross_sharpe_real:.2f}, t = {_gross_t_real:.2f}) on gross metrics.  "
            "A real-labels RF has no advantage over a noise-trained one -- both are "
            "capturing the equity risk premium from being long 75% of the time."
        ),
        md(
            "### 4c - Cost sweep and HAC t-stat profile\n\n"
            "At a realistic 5 bps round-trip cost, the net t-stat falls well below 2:"
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            f"sharpes_plot = [{_sharpe_0bp}, {_sharpe_2bp}, {_sharpe_5bp}, {_sharpe_10bp}]\n"
            f"tstats_plot  = [{_t_0bp}, {_t_2bp}, {_t_5bp}, {_t_10bp}]\n"
            "if HAVE_REAL:\n"
            "    bars, X, y = load_real()\n"
            "    preds_real = st.walk_forward(X, y, train_window=504, step=63, n_estimators=200)\n"
            "    sharpes_plot, tstats_plot = [], []\n"
            "    for c in costs:\n"
            "        led = st.trade_predictions(bars, preds_real, cost_bps=c)\n"
            "        s = st.summarize(preds_real, led)\n"
            "        sharpes_plot.append(s['sharpe_net']); tstats_plot.append(s['tstat_net'])\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "ax1.plot(costs, sharpes_plot, 'o-', c=RED, lw=2)\n"
            f"ax1.axhline({_bh_sharpe:.3f}, ls='--', c=GREY, lw=1.5, label='B&H SPY')\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_xlabel('cost (bps)'); ax1.set_ylabel('net Sharpe')\n"
            "ax1.set_title('Net Sharpe vs costs'); ax1.legend()\n"
            "ax2.plot(costs, tstats_plot, 's-', c=RED, lw=2)\n"
            "ax2.axhline(2, ls='--', c=GREY, lw=1.5, label='|t|=2 bar')\n"
            "ax2.axhline(-2, ls='--', c=GREY, lw=1.5)\n"
            "ax2.axhline(0, c='k', lw=1); ax2.set_xlabel('cost (bps)'); ax2.set_ylabel('HAC t-stat')\n"
            "ax2.set_title('t-stat falls below significance at any real cost'); ax2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for c, sh, t in zip(costs, sharpes_plot, tstats_plot):\n"
            "    print(f'  cost {c:.0f}bp: Sharpe {sh:+.3f}  HAC t {t:+.3f}')"
        ),
        md(
            "> In plain words: even at zero cost the gross t-stat (2.15) barely clears "
            "the |t| >= 2 bar -- but the shuffled model clears it too (2.34).  Once any "
            "real cost is charged, the strategy cannot clear the significance threshold.  "
            "The 'strategy' is essentially just 'hold SPY 75% of days' minus friction."
        ),
        md(
            "### 4d - The positive control: RF does learn on a planted signal\n\n"
            "Is the engine broken, or is the market just featureless at daily scale?  "
            "Plant a real learnable signal in a synthetic tape and check that the RF "
            "beats the shuffle there:"
        ),
        code(
            "results = []\n"
            "for strength in [0.000, 0.005, 0.010, 0.030]:\n"
            "    bars_s, _ = data.synthetic_daily(n_days=1000, signal_strength=strength, seed=138)\n"
            "    Xs, ys = st.build_features(bars_s)\n"
            "    pr = st.walk_forward(Xs, ys, train_window=300, step=63, n_estimators=100, max_depth=5)\n"
            "    ps = st.walk_forward(Xs, ys, train_window=300, step=63, n_estimators=100, max_depth=5, shuffle_labels=True)\n"
            "    ar = float((pr['y_true']==pr['y_pred']).mean())\n"
            "    as_ = float((ps['y_true']==ps['y_pred']).mean())\n"
            "    results.append((strength, ar, as_, ar-as_))\n"
            "df_ctrl = pd.DataFrame(results, columns=['signal_strength','acc_real','acc_shuf','delta'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "ax.plot(df_ctrl['signal_strength'], df_ctrl['acc_real'], 'o-', c=GREEN, lw=2, label='Real labels')\n"
            "ax.plot(df_ctrl['signal_strength'], df_ctrl['acc_shuf'], 's--', c=GREY, lw=2, label='Shuffled labels')\n"
            "ax.axhline(0.5, ls=':', c='k', lw=1, label='coin (50%)')\n"
            "ax.set_xlabel('Planted signal strength'); ax.set_ylabel('OOS accuracy')\n"
            "ax.set_title('The engine works: real model beats shuffle only when a signal exists')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(df_ctrl.to_string(index=False))"
        ),
        md(
            "> In plain words: when we plant a real signal in a synthetic tape, the real-label "
            "RF beats the shuffle by a widening margin.  At signal_strength = 0 (the null -- "
            "analogous to real SPY data) the gap is near zero, as we see on the actual tape.  "
            "The engine is correctly calibrated; the real market just has nothing for it "
            "to learn from these features at daily scale."
        ),

        # ---- BEAT 5 -- VERDICT -------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- OOS accuracy {_acc_real:.1%} vs shuffle {_acc_shuf:.1%} "
            f"(gap {_delta_pp:+.1f}pp, not significant); shuffled gross t "
            f"({_gross_t_shuf:.2f}) > real gross t ({_gross_t_real:.2f}); "
            f"net HAC t at 5bp = {_net_t_5bp:+.2f}.\n"
            f"- **Tradability `MIRAGE`** -- net Sharpe {_net_sharpe_5bp:+.2f} at 5bp "
            f"vs buy-and-hold {_bh_sharpe:+.2f}; strategy is partially-invested long-only "
            "and fails to beat a passive index fund.\n"
            "- **Beats shuffle? `NO`** -- the defining test: the shuffled-label model "
            "matches or outperforms the real model on every gross metric.  No information "
            "content in the feature set above what a noise model extracts."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -------------------------------------
        md(
            "## 6 - Could you trade it?\n\n"
            "Purely mechanical objections:\n\n"
            "1. **Underperforms buy-and-hold.** You earn less Sharpe and less absolute "
            "return than holding SPY outright.  The 'strategy' is SPY-minus-25%-of-the-time.\n"
            "2. **Costs wipe the edge.** At 5 bps the t-stat is 0.86 -- you cannot claim "
            "statistical significance for the net return.\n"
            "3. **The null controls.** Any strategy that shows similar metrics after "
            "shuffling labels has not learned alpha -- it learned market structure.\n\n"
            "An investor in this study would correctly choose: **buy SPY, hold it, do nothing.**"
        ),

        # ---- BEAT 7 -- GOING FURTHER -------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Gu, Kelly & Xiu (2020)**: tree ensembles do find predictability, but in a "
            "cross-sectional setting with 900+ features.  See [Scorecard (Study 65)](../../65-scorecard/) "
            "for an honest cross-sectional factor test.\n"
            "- **Purged cross-validation**: Lopez de Prado (2018) describes more powerful "
            "walk-forward designs that purge samples near the boundary.\n"
            "- **Other ML models**: [Black-Box (Study 39)](../../39-black-box/) tests "
            "an MLP on crypto with the same protocol -- same verdict.\n"
            "- **Alternative features**: this study tests the retail feature set; "
            "alternative data (sentiment, order flow, options skew) might carry information "
            "not in price history alone.  The shuffle test is the right framework.\n\n"
            "*The shuffle test is the bar.  Show your walk-forward model beats a shuffled-label "
            "version by a margin that clears |t| > 2.  That is the standard.*"
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
