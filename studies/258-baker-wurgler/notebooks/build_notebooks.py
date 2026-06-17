"""Generate the two narrative notebooks for Study 258 (Baker-Wurgler).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the repo-level
^GSPC / ^RUT caches if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_months=737, window_start="1965-02", window_end="2026-06",
    fp_sent="d88ba1009823", fp_ret="24206d518a4c",
    # regime sort (GSPC next-month, by prior sentiment tertile)
    low_ann=8.78, low_t=2.36, low_n=246,
    mid_ann=9.24, mid_t=3.41, mid_n=245,
    high_ann=7.16, high_t=2.22, high_n=246,
    # predictive regression
    beta_bps=-31.01, beta_t=-1.29, r2=0.0025,
    # low-minus-high spread
    lmh_ann=0.81, lmh_t=0.32,
    lmh_sr_lo=-0.257, lmh_sr_hi=0.356, lmh_frac_neg=0.385,
    # SMB cross-sectional leg (^RUT - ^GSPC, 1987-2026)
    smb_low_ann=1.66, smb_high_ann=-1.83, smb_beta_bps=-17.33, smb_beta_t=-0.77, smb_n=465,
    # sub-periods (gspc regression)
    sub1_beta_t=-1.39, sub1_low=9.1, sub1_high=0.7, sub1_n=299,   # 1965-1989
    sub2_beta_t=-0.49, sub2_low=14.0, sub2_high=9.9, sub2_n=216,  # 1990-2007
    sub3_beta_t=-0.16, sub3_low=7.3, sub3_high=14.1, sub3_n=222,  # 2008-2026 (sign flips!)
    # timing overlay vs buy-and-hold
    ov_net_ann=2.9, ov_net_sr=0.32, ov_net_dd=-0.458,
    bh_ann=8.39, bh_sr=0.56, bh_dd=-0.526,
    ov_switch_yr=0.73, ov_cost_bps=220.0,
    ov_short_ann=0.47, ov_short_sr=0.04, ov_short_dd=-0.644,
    # synthetic positive control
    syn_beta_bps=-52.44, syn_beta_t=-2.58, null_beta_t=0.23,
)


# ---------------------------------------------------------------------------
# Shared analysis preamble
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

from baker_wurgler import data, strategy as st

HAVE_REAL = os.path.exists(data.GSPC_CACHE)
sent = data.bw_monthly()

if HAVE_REAL:
    mkt = data.fetch_market_monthly(fetch=False)
    df = data.merge_sentiment_returns(mkt, sent)
    print(f"Real tape loaded: {df.shape[0]} return-months, "
          f"{df.index[0].strftime('%Y-%m')} -> {df.index[-1].strftime('%Y-%m')}")
else:
    mkt = df = None
    print("No ^GSPC cache -- frozen headline numbers from R dict will be used")
"""

R_INJECT = f"""
# Frozen headline numbers (mirror of docs/results.md, as-of 2026-06-17)
R = {R!r}
"""


# ---------------------------------------------------------------------------
# Notebook 1 -- For the curious
# ---------------------------------------------------------------------------
def build_curious() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 258 -- Baker-Wurgler Investor Sentiment
## For the curious: does Wall Street's mood forecast the market?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

Baker & Wurgler (2006) built a famous **investor-sentiment index** from six market
proxies (the closed-end-fund discount, IPO volume and first-day pops, trading turnover,
the share of equity in new issues, and the dividend premium).  Their claim is
**contrarian**: when sentiment is *high* (everyone is euphoric), future returns are
*low*; when sentiment is *low* (everyone is fearful), future returns are *high*.

And the effect is supposed to be strongest for **speculative, hard-to-value stocks** --
small caps, young firms, non-dividend payers -- the ones most prone to a sentiment-driven
mispricing that later reverses.

We hardcode a reconstruction of the BW index and ask the simplest possible question:
**sort the market's next-month return by the prior month's sentiment regime -- do
low-sentiment months really beat high-sentiment months?**
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The sentiment tape

Here is the reconstructed Baker-Wurgler index.  The peaks are the famous bubbles --
the late-1960s electronics/conglomerate craze, the 1980-81 froth, the 2000 dot-com top,
the 2007 pre-GFC high, and the 2021 meme-stock blow-off.  The troughs are the bottoms
of 1974, 1991, 2003 and 2009.
"""),
        code("""\
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(sent.index, sent.values, color=GREY, lw=1.3)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.fill_between(sent.index, sent.values, 0, where=(sent.values > 0), color=RED, alpha=0.25, label="optimistic")
ax.fill_between(sent.index, sent.values, 0, where=(sent.values <= 0), color=GREEN, alpha=0.25, label="pessimistic")
ax.set_title("Reconstructed Baker-Wurgler investor-sentiment index (standardized)")
ax.set_ylabel("Sentiment (std units)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../docs/sentiment_index.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Index window: {sent.index[0].strftime('%Y-%m')} -> {sent.index[-1].strftime('%Y-%m')}")
"""),

        md("## The contrarian test: next-month return by sentiment regime"),
        code("""\
months_ann = lambda x: x * 100
if HAVE_REAL:
    reg = st.regime_sort(df, "ret", n_bins=3)
    low_ann  = reg.loc["low",  "mean_ann"] * 100
    mid_ann  = reg.loc["mid",  "mean_ann"] * 100
    high_ann = reg.loc["high", "mean_ann"] * 100
else:
    low_ann, mid_ann, high_ann = R["low_ann"], R["mid_ann"], R["high_ann"]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(["Low\\nsentiment", "Mid", "High\\nsentiment"],
              [low_ann, mid_ann, high_ann],
              color=[GREEN, GREY, RED])
ax.axhline(0, color="black", lw=0.8)
ax.set_title("S&P 500 next-month return by prior sentiment regime (%/yr)")
ax.set_ylabel("Annualised next-month return (%/yr)")
for b, v in zip(bars, [low_ann, mid_ann, high_ann]):
    ax.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("../docs/regime_bars.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Low-sentiment months:  {low_ann:.2f}%/yr")
print(f"High-sentiment months: {high_ann:.2f}%/yr")
print(f"Low beats High by {low_ann - high_ann:+.2f} pp/yr -- the BW contrarian SIGN is right...")
print(f"...but the low-minus-high spread is only +{R['lmh_ann']:.2f}%/yr at HAC t = {R['lmh_t']:.2f} (not significant).")
"""),

        md("""\
## The verdict for the curious: right idea, wrong magnitude (on the index)

The sign is exactly what Baker & Wurgler predict -- **low-sentiment months
(+{:.1f}%/yr) outperform high-sentiment months (+{:.1f}%/yr)**.  The instinct is real.

But on the *aggregate S&P 500 index*, the gap is tiny and statistically indistinguishable
from luck (HAC *t* = {:.2f} on the low-minus-high spread).  Two honest reasons:

1. **BW lives in the cross-section, not the index.** The published effect is about
   *speculative stocks vs safe stocks* -- small caps, young firms, non-payers.  The
   plain S&P 500 is dominated by exactly the safe, easy-to-value names where the
   sentiment mispricing is weakest.

2. **The effect has decayed.** It was strongest in the 1965-1989 sample and has faded
   since -- in the most recent sub-period the sign even *flips* (we show this for the
   quants).  Post-publication arbitrage is the usual suspect.
""".format(R["low_ann"], R["high_ann"], R["lmh_t"])),

        md("## A tradable timing rule? Long when fearful, flat when greedy"),
        code("""\
if HAVE_REAL:
    ov = st.timing_overlay(df, "ret", n_bins=3, short_high=False, one_way_bps=5.0)
    net = ov["net"]; bh = ov["buyhold"]
    cum_net = (1 + net).cumprod(); cum_bh = (1 + bh).cumprod()
    s_net = st.summarize(net); s_bh = st.summarize(bh)
    net_ann = s_net["mean"]*12*100; bh_ann = s_bh["mean"]*12*100
else:
    net_ann, bh_ann = R["ov_net_ann"], R["bh_ann"]
    cum_net = cum_bh = None

fig, ax = plt.subplots(figsize=(11, 5))
if cum_net is not None:
    ax.plot(cum_bh.index, cum_bh.values, color=GREY, lw=1.6, label=f"Buy & hold ({bh_ann:.1f}%/yr)")
    ax.plot(cum_net.index, cum_net.values, color=RED, lw=1.6, label=f"Sentiment timing, net ({net_ann:.1f}%/yr)")
    ax.set_yscale("log")
    ax.set_title("Long-when-fearful timing rule vs buy-and-hold (log scale, net of costs)")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=10)
else:
    ax.text(0.5, 0.5, f"Cache absent\\nTiming {R['ov_net_ann']:.1f}%/yr  vs  Buy&hold {R['bh_ann']:.1f}%/yr",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color=RED)
plt.tight_layout()
plt.savefig("../docs/timing_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Timing rule (long in low-sentiment months, flat otherwise), net: {R['ov_net_ann']:.1f}%/yr")
print(f"Unconditional buy-and-hold:                                    {R['bh_ann']:.1f}%/yr")
print(f"-> The timing rule LOSES to buy-and-hold by {R['bh_ann']-R['ov_net_ann']:.1f} pp/yr.")
print(f"   Sitting out the high-sentiment months also sits out a lot of upside.")
"""),

        md("""\
## The bottom line for the curious

Baker-Wurgler captures a *real psychological instinct* -- buy fear, sell greed -- and the
sign of the effect shows up faintly even on the broad S&P 500.  But:

- The aggregate-index spread is **not statistically significant** (HAC *t* = {:.2f}).
- A naive **timing rule loses to buy-and-hold** ({:.1f}%/yr vs {:.1f}%/yr net) -- the
  fearful months you'd sit out contain too much of the market's long-run return.
- The cross-sectional leg (small-minus-large) has the right sign but is also
  insignificant on our proxy.

This is a **Weak / Mirage** result: the literature is real and the sign is right, but
there is no robust, tradable edge here on the aggregate index.
""".format(R["lmh_t"], R["ov_net_ann"], R["bh_ann"])),

        md("*Desk verdict: **Weak / Mirage** -- see [README.md](../README.md) and [docs/results.md](../docs/results.md) for full numbers.*"),
    ]
    _write(nb, "01_for_the_curious.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2 -- For the quants
# ---------------------------------------------------------------------------
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 258 -- Baker-Wurgler Investor Sentiment
## For the quants: predictive regression, HAC tests, sub-period decay, synthetic control

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the engine recovers a planted contrarian effect

Before touching the real tape we confirm the machinery is honest.  ``synthetic_market``
plants a *contrarian* relationship (high prior sentiment -> low next-month return) with
a knob ``contrarian_bps``.  ``contrarian_bps = 0`` is the null.
"""),
        code("""\
syn, truth = data.synthetic_market(contrarian_bps=60.0, seed=258)
pr_syn = st.predictive_regression(syn, "ret")
print(f"Synthetic (contrarian_bps=60): beta = {pr_syn['beta']*1e4:+.2f} bps/unit  HAC t = {pr_syn['beta_t']:+.2f}")

syn0, _ = data.synthetic_market(contrarian_bps=0.0, seed=258)
pr_null = st.predictive_regression(syn0, "ret")
print(f"Null      (contrarian_bps=0):  beta = {pr_null['beta']*1e4:+.2f} bps/unit  HAC t = {pr_null['beta_t']:+.2f}")
print("-> Engine reads a clean negative (contrarian) slope when planted, ~0 on the null.")
"""),

        md("## Real tape: regime sort and predictive regression"),
        code("""\
if HAVE_REAL:
    reg = st.regime_sort(df, "ret", n_bins=3)
    print("=== Next-month S&P 500 return by prior sentiment tertile ===")
    for lab in ["low", "mid", "high"]:
        r = reg.loc[lab]
        print(f"  {lab:4s}: {r['mean_ann']*100:+6.2f}%/yr  vol={r['vol_ann']*100:4.1f}%  HAC t={r['tstat']:+.2f}  n={int(r['n'])}")
    pr = st.predictive_regression(df, "ret")
    lmh = st.low_minus_high(df, "ret", n_bins=3)
    s_lmh = st.summarize(lmh)
    print()
    print(f"Predictive regression slope: {pr['beta']*1e4:+.2f} bps per sentiment unit  HAC t = {pr['beta_t']:+.2f}  R2 = {pr['r2']:.4f}")
    print(f"Low-minus-high spread: {s_lmh['mean']*12*100:+.2f}%/yr  HAC t = {s_lmh['tstat']:+.2f}")
else:
    print(f"Frozen: low={R['low_ann']:+.2f}%/yr (t={R['low_t']}) mid={R['mid_ann']:+.2f} high={R['high_ann']:+.2f}%/yr (t={R['high_t']})")
    print(f"Frozen: regression beta={R['beta_bps']:+.2f} bps/unit  HAC t={R['beta_t']:+.2f}  R2={R['r2']}")
    print(f"Frozen: low-minus-high {R['lmh_ann']:+.2f}%/yr  HAC t={R['lmh_t']:+.2f}")
print()
print("The slope is NEGATIVE (the Baker-Wurgler contrarian sign) but |t| < 2 -- not significant.")
"""),

        md("## Block-bootstrap CI on the low-minus-high spread Sharpe"),
        code("""\
if HAVE_REAL:
    lmh = st.low_minus_high(df, "ret", n_bins=3)
    try:
        from quantlab.stats import sharpe_ci_bootstrap
        ci = sharpe_ci_bootstrap(lmh, n_boot=2000, periods_per_year=12, seed=258)
        print(f"Spread Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% of resamples negative)")
    except ImportError:
        print(f"[quantlab unavailable -- frozen CI: [{R['lmh_sr_lo']:+.3f}, {R['lmh_sr_hi']:+.3f}]]")
else:
    print(f"Frozen Sharpe 95% CI: [{R['lmh_sr_lo']:+.3f}, {R['lmh_sr_hi']:+.3f}]  ({R['lmh_frac_neg']*100:.0f}% negative)")
print("The CI straddles zero -- the spread is statistically indistinguishable from luck.")
"""),

        md("## Sub-period decay: the effect fades and then *flips* sign"),
        code("""\
periods = [("1965-1989", "1965", "1989"), ("1990-2007", "1990", "2007"), ("2008-2026", "2008", "2026")]
if HAVE_REAL:
    rows = []
    for label, a, b in periods:
        sub = df.loc[a:b]
        pr = st.predictive_regression(sub, "ret")
        reg = st.regime_sort(sub, "ret", n_bins=3)
        lo = reg.loc["low", "mean_ann"]*100 if "low" in reg.index else np.nan
        hi = reg.loc["high", "mean_ann"]*100 if "high" in reg.index else np.nan
        rows.append({"period": label, "beta_t": pr["beta_t"], "low": lo, "high": hi, "n": pr["n"]})
        print(f"  {label}: beta HAC t={pr['beta_t']:+.2f}  low={lo:+.1f}%/yr  high={hi:+.1f}%/yr  n={pr['n']}")
    sub_df = pd.DataFrame(rows)
else:
    sub_df = pd.DataFrame([
        {"period": "1965-1989", "beta_t": R["sub1_beta_t"], "low": R["sub1_low"], "high": R["sub1_high"], "n": R["sub1_n"]},
        {"period": "1990-2007", "beta_t": R["sub2_beta_t"], "low": R["sub2_low"], "high": R["sub2_high"], "n": R["sub2_n"]},
        {"period": "2008-2026", "beta_t": R["sub3_beta_t"], "low": R["sub3_low"], "high": R["sub3_high"], "n": R["sub3_n"]},
    ])
    for _, r in sub_df.iterrows():
        print(f"  {r['period']}: beta HAC t={r['beta_t']:+.2f}  low={r['low']:+.1f}%/yr  high={r['high']:+.1f}%/yr  n={int(r['n'])}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
colors = [GREEN if t <= -2 else AMBER if t <= -1 else RED for t in sub_df["beta_t"]]
axes[0].bar(sub_df["period"], sub_df["beta_t"], color=colors)
axes[0].axhline(-2.0, color="black", ls="--", lw=0.9, label="t=-2 bar")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("Predictive-regression HAC t by sub-period (negative = contrarian works)")
axes[0].set_ylabel("HAC t-stat (slope)")
axes[0].legend(fontsize=9)
w = 0.35
x = np.arange(len(sub_df))
axes[1].bar(x - w/2, sub_df["low"], w, color=GREEN, label="low-sentiment months")
axes[1].bar(x + w/2, sub_df["high"], w, color=RED, label="high-sentiment months")
axes[1].set_xticks(x); axes[1].set_xticklabels(sub_df["period"])
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("Next-month return by regime, per sub-period (%/yr)")
axes[1].set_ylabel("%/yr"); axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig("../docs/subperiods.png", dpi=120, bbox_inches="tight")
plt.show()
print("\\n1965-1989 had the strongest contrarian tilt (t={:.2f}); by 2008-2026 the sign FLIPS".format(R["sub1_beta_t"]))
print("  (high-sentiment months {:.1f}%/yr actually BEAT low-sentiment {:.1f}%/yr) -- classic post-publication decay.".format(R["sub3_high"], R["sub3_low"]))
"""),

        md("## The cross-sectional leg: small-minus-large (^RUT - ^GSPC)"),
        code("""\
if HAVE_REAL and "smb" in df.columns:
    dfs = df.dropna(subset=["smb"])
    reg_s = st.regime_sort(dfs, "smb", n_bins=3)
    pr_s = st.predictive_regression(dfs, "smb")
    print(f"SMB window: {dfs.index[0].strftime('%Y-%m')} -> {dfs.index[-1].strftime('%Y-%m')}  n={len(dfs)}")
    for lab in ["low", "mid", "high"]:
        if lab in reg_s.index:
            print(f"  {lab:4s}: SMB {reg_s.loc[lab,'mean_ann']*100:+.2f}%/yr  HAC t={reg_s.loc[lab,'tstat']:+.2f}")
    print(f"SMB regression slope: {pr_s['beta']*1e4:+.2f} bps/unit  HAC t={pr_s['beta_t']:+.2f}")
else:
    print(f"Frozen SMB: low={R['smb_low_ann']:+.2f}%/yr  high={R['smb_high_ann']:+.2f}%/yr")
    print(f"Frozen SMB regression: {R['smb_beta_bps']:+.2f} bps/unit  HAC t={R['smb_beta_t']:+.2f}  n={R['smb_n']}")
print()
print("BW says small/speculative names suffer MOST after high sentiment.")
print("The sign is right (small-minus-large is positive after low sentiment, negative after high)")
print("but the proxy effect is again insignificant (|t| < 1) on the 1987-2026 window.")
"""),

        md("## Tradability: the timing overlay and its costs"),
        code("""\
if HAVE_REAL:
    ov = st.timing_overlay(df, "ret", n_bins=3, short_high=False, one_way_bps=5.0)
    s_net = st.summarize(ov["net"]); s_bh = st.summarize(ov["buyhold"])
    tc = st.turnover_count(ov)
    print("=== Long-when-low-sentiment, flat-otherwise (net of 5bps one-way x NAV) ===")
    print(f"  Net:        {s_net['mean']*12*100:+.2f}%/yr  SR(ann)={s_net['sharpe']*12**0.5:+.2f}  maxDD={s_net['max_drawdown']:.1%}")
    print(f"  Buy & hold: {s_bh['mean']*12*100:+.2f}%/yr  SR(ann)={s_bh['sharpe']*12**0.5:+.2f}  maxDD={s_bh['max_drawdown']:.1%}")
    print(f"  Switches/yr: {tc['switches_per_year']:.2f}  total cost: {tc['total_cost_bps']:.1f} bps")
    ov_s = st.timing_overlay(df, "ret", n_bins=3, short_high=True, one_way_bps=5.0)
    s_short = st.summarize(ov_s["net"])
    print(f"  Short-the-high variant (borrow paid): {s_short['mean']*12*100:+.2f}%/yr  SR={s_short['sharpe']*12**0.5:+.2f}  maxDD={s_short['max_drawdown']:.1%}")
else:
    print(f"Frozen: timing net={R['ov_net_ann']:+.2f}%/yr SR={R['ov_net_sr']} vs buy-hold {R['bh_ann']:+.2f}%/yr SR={R['bh_sr']}")
    print(f"Frozen: switches/yr={R['ov_switch_yr']}  short-variant {R['ov_short_ann']:+.2f}%/yr SR={R['ov_short_sr']}")
print()
print("The timing rule UNDERPERFORMS buy-and-hold on both return and Sharpe.")
print("Even with only ~0.7 switches/yr (low turnover), sitting out greedy months costs too much upside.")
print("Shorting the high-sentiment months is worse still (pays borrow, eats the equity risk premium).")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 258 -- Baker-Wurgler Investor Sentiment ===")
print()
print(f"Signal: WEAK")
print(f"  Sign is correct (contrarian): low-sentiment {R['low_ann']:.1f}%/yr > high-sentiment {R['high_ann']:.1f}%/yr")
print(f"  BUT predictive slope HAC t = {R['beta_t']:.2f} (|t| < 2); low-minus-high spread t = {R['lmh_t']:.2f}")
print(f"  Bootstrap Sharpe CI [{R['lmh_sr_lo']:+.3f}, {R['lmh_sr_hi']:+.3f}] straddles zero")
print(f"  Effect concentrated in 1965-1989 (t={R['sub1_beta_t']:.2f}); sign FLIPS post-2008")
print(f"  Cross-sectional SMB leg right-signed but insignificant (t={R['smb_beta_t']:.2f})")
print(f"  Literature support is strong (BW 2006/2007) -> WEAK, not None")
print()
print(f"Tradability: MIRAGE")
print(f"  Timing overlay {R['ov_net_ann']:.1f}%/yr net LOSES to buy-and-hold {R['bh_ann']:.1f}%/yr")
print(f"  Lower Sharpe ({R['ov_net_sr']} vs {R['bh_sr']}); short variant worse ({R['ov_short_ann']:.1f}%/yr)")
print(f"  No tradable edge on the aggregate index beyond passive buy-and-hold")
print()
print("Price-only returns (no dividends) -- NAMED on the Signal axis.")
print("Hardcoded index is a reconstruction -- NAMED; verbatim BW file preserves regime structure.")
print()
print("Bottom line: Weak/Mirage -- a real psychological instinct and a real")
print("  academic literature, but no robust, tradable edge on the broad market index.")
"""),
    ]
    _write(nb, "02_for_the_quants.ipynb")


# ---------------------------------------------------------------------------
# Shared _write helper
# ---------------------------------------------------------------------------
def _write(nb: nbf.NotebookNode, filename: str) -> None:
    path = os.path.join(HERE, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Execute notebooks with nbconvert
# ---------------------------------------------------------------------------
def _execute(filename: str) -> None:
    import subprocess, sys
    path = os.path.join(HERE, filename)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {filename}:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"nbconvert failed for {filename}")
    print(f"Executed {filename}")


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("Done.")
