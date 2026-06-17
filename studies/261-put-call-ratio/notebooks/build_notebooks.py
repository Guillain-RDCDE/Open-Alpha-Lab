"""Generate the two narrative notebooks for Study 261 (Put-Call-Ratio).

    python notebooks/build_notebooks.py

This script writes both notebooks and then executes them in-place with nbconvert
(no --allow-errors). The synthetic cells run anywhere, offline and deterministic;
the real-tape cells use the cached ^GSPC monthly series under ../_cache/ if
present, and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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
    n_months=276, n_active=240, fingerprint="8edbde520f70",
    pcr_mean=0.913, pcr_std=0.107,
    # continuous predictive regression of next-month ret on expanding pcr z-score
    reg_slope=-0.0007, reg_t=-0.16, reg_r2=0.0002, reg_n=240,
    # extreme-fear timing rule (q=0.80) vs buy-and-hold
    in_mkt_frac=0.262,
    strat_ann=2.18, strat_t=1.02, strat_sr=0.20, strat_hit=0.163, strat_dd=-0.371,
    strat_net_ann=2.02, strat_net_t=0.95,
    bh_ann=9.63, bh_t=2.81, bh_sr=0.64, bh_dd=-0.526,
    n_switches=64,
    # conditional next-month return: fear months vs calm months
    fear_n=63, fear_mean=0.693, fear_t=1.02,
    calm_n=177, calm_mean=0.842, calm_t=3.11,
    fear_minus_calm=-0.149, welch_t=-0.18, welch_p=0.856,
    # random-timing permutation
    perm_p=0.615, rand_mean=0.815,
    # sub-periods (strat ann %/yr, t, n ; bh ann)
    sub1_lab="2006-2012", sub1=-2.0, sub1_t=-0.41, sub1_n=84, sub1_bh=3.7,
    sub2_lab="2013-2019", sub2=3.5, sub2_t=1.91, sub2_n=84, sub2_bh=11.6,
    sub3_lab="2020-2025", sub3=5.5, sub3_t=1.69, sub3_n=72, sub3_bh=14.3,
    # synthetic positive control (predictive regression slope t-stat)
    syn_null_t=-0.95, syn_live_t=6.98, syn_strong_t=14.81,
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

from put_call_ratio import data, strategy as st

CACHE_PATH = data.GSPC_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    panel = data.build_real_panel(fetch=False)
    sig = st.extreme_signal(panel["pcr"], q=0.80)
    tr = st.timing_returns(panel, q=0.80, one_way_bps=5.0)
    print(f"Real panel loaded: {len(panel)} months  {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"Active (post-burn-in) months: {len(tr)}")
else:
    panel = sig = tr = None
    print("No real ^GSPC cache -- frozen headline numbers from R dict will be used")
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
# Study 261 -- Put-Call-Ratio
## For the curious: does the CBOE put/call ratio time market bottoms?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

When traders get scared, they buy puts. The **CBOE total put/call ratio** -- puts
traded divided by calls traded -- is the classic "fear gauge by flow". The
folklore is irresistibly contrarian: when everyone is buying puts (ratio spikes
high), the crowd is *too* fearful, the selling is exhausted, and the market is
near a **bottom** -- so a high put/call ratio is a *buy*.

We test the simplest honest version: read the ratio at month-end, and if it is in
its top quintile of history (extreme fear), hold the S&P 500 for the next month;
otherwise sit in cash. Does buying the fear beat just owning the index?
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The put/call ratio and where it spikes

The ratio averages around **0.92** and spikes above ~1.15 in genuine panics:
October 2008 (Lehman), December 2018 (the Powell-pivot selloff), March 2020
(COVID crash). Those spikes are exactly the months the contrarian story says you
should be buying.
"""),
        code("""\
fig, ax = plt.subplots(figsize=(11, 4))
pcr = data.pcr_series()
ax.plot(pcr.index, pcr.values, color=GREY, lw=1.2)
thr = pcr.quantile(0.80)
ax.axhline(thr, color=RED, ls="--", lw=1.0, label=f"80th pct = {thr:.2f} (extreme fear)")
ax.axhline(pcr.mean(), color="black", ls=":", lw=0.8, label=f"mean = {pcr.mean():.2f}")
fear = pcr[pcr >= thr]
ax.scatter(fear.index, fear.values, color=RED, s=18, zorder=5, label="extreme-fear months")
ax.set_title("CBOE total put/call ratio (month-end, hardcoded)")
ax.set_ylabel("put/call ratio")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("../docs/pcr_series.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Mean put/call = {pcr.mean():.3f}, std = {pcr.std():.3f}")
"""),

        md("## Buying the fear vs owning the index"),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
if HAVE_REAL:
    cum_strat = (1 + tr["strat"]).cumprod()
    cum_bh    = (1 + tr["bh"]).cumprod()
    ax.plot(cum_bh.index,    cum_bh.values,    color=GREY,  lw=1.6, label="Buy & hold S&P 500 (price-only)")
    ax.plot(cum_strat.index, cum_strat.values, color=RED,   lw=1.8, label="Contrarian extreme-fear timing")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 -- buying the fear lags badly (log scale)")
    ax.set_ylabel("Cumulative ($, log)")
    ax.legend(fontsize=9)
else:
    ax.text(0.5, 0.5,
            f"Cache absent\\nFrozen: strat = +{R['strat_ann']:.1f}%/yr  vs  buy-hold +{R['bh_ann']:.1f}%/yr",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color=RED)
    ax.set_title("(frozen numbers)")
plt.tight_layout()
plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Extreme-fear timing: +{R['strat_ann']:.1f}%/yr  (invested only {R['in_mkt_frac']*100:.0f}% of months)")
print(f"Buy & hold        : +{R['bh_ann']:.1f}%/yr")
print(f"-> Sitting in cash 3/4 of the time, you miss most of the bull market.")
"""),

        md("""\
## The killer: extreme-fear months are NOT better than calm months

If the contrarian story were true, the month *after* an extreme put/call spike
would beat an ordinary month. It does not -- if anything it is slightly *worse*.
"""),
        code("""\
if HAVE_REAL:
    s = sig.iloc[36:]
    fwd = panel["fwd_ret"].iloc[36:]
    fear = fwd[s == 1]; calm = fwd[s == 0]
    fear_mean, calm_mean = fear.mean()*100, calm.mean()*100
    fear_n, calm_n = len(fear), len(calm)
else:
    fear_mean, calm_mean = R["fear_mean"], R["calm_mean"]
    fear_n, calm_n = R["fear_n"], R["calm_n"]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(["After extreme fear", "After calm"], [fear_mean, calm_mean],
              color=[RED, GREEN])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Mean next-month S&P return (%)")
ax.set_title("Next-month return by put/call state")
for b, v, n in zip(bars, [fear_mean, calm_mean], [fear_n, calm_n]):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:+.2f}%\\n(n={n})", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("../docs/by_state.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"After extreme fear: {fear_mean:+.2f}%/mo (n={fear_n})")
print(f"After calm        : {calm_mean:+.2f}%/mo (n={calm_n})")
print(f"Difference (fear - calm): {R['fear_minus_calm']:+.2f}%/mo  Welch t = {R['welch_t']:+.2f}  p = {R['welch_p']:.2f}")
print("-> The 'fear bottom' edge is absent (and the wrong sign) on the real tape.")
"""),

        md("""\
## The bottom line for the curious

The put/call ratio is a real, well-measured fear gauge -- it spikes exactly where
you'd expect (2008, 2018, 2020). But as a **market-timing buy signal** it fails:

- The month after an extreme-fear spike returns **+0.69%**, *below* the +0.84% of
  a calm month (Welch *t* = -0.18 -- statistically indistinguishable, wrong sign).
- The timing rule earns **+2.2%/yr** versus the index's **+9.6%/yr**, because
  buying only the fear keeps you in cash ~74% of the time and out of the bull run.
- A *random* selection of the same number of in-market months beats the
  fear-timed selection 61% of the time.

There is no contrarian bottom-timing edge here. **Signal: None. Tradability: Mirage.**
"""),

        md("*Desk verdict: **None / Mirage** -- see [README.md](../README.md) and [docs/results.md](../docs/results.md) for full numbers.*"),
    ]

    _write(nb, "01_for_the_curious.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2 -- For the quants
# ---------------------------------------------------------------------------
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 261 -- Put-Call-Ratio
## For the quants: predictive regression, HAC tests, random-timing null, sub-periods

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the engine finds a contrarian edge when one is planted

`data.synthetic_panel(contrarian_beta=b)` plants `fwd_ret = base + b*(pcr-mean) + noise`.
A positive `b` is the folklore (high fear precedes high returns). The predictive
regression must recover it at `b>0` and read ~zero at `b=0`.
"""),
        code("""\
for beta, lab in [(0.0, "null"), (0.15, "planted"), (0.30, "strong")]:
    dfx, _ = data.synthetic_panel(contrarian_beta=beta, seed=261)
    reg = st.predictive_regression(dfx)
    print(f"  beta={beta:<4} ({lab:<7}): slope={reg['slope']:+.4f}  HAC t={reg['tstat']:+.2f}  R2={reg['r2']:.3f}")
print()
print("-> Engine reads ~0 on the null and a strongly positive slope when fear is")
print("   genuinely informative. The machinery is truthful.")
"""),

        md("## Real tape: the continuous predictive regression is flat"),
        code("""\
if HAVE_REAL:
    reg = st.predictive_regression(panel)
    print("=== Next-month S&P return on expanding put/call z-score (HAC) ===")
    print(f"  slope = {reg['slope']:+.4f} per 1-sigma of put/call")
    print(f"  HAC t = {reg['tstat']:+.2f}")
    print(f"  R^2   = {reg['r2']:.4f}   n = {reg['n']}")
else:
    print(f"Frozen: slope={R['reg_slope']:+.4f}  HAC t={R['reg_t']:+.2f}  R2={R['reg_r2']:.4f}  n={R['reg_n']}")
print()
print("A contrarian edge would show a POSITIVE slope (high fear -> high fwd return).")
print(f"The real slope is essentially zero (t = {R['reg_t']:+.2f}) -- no predictive content.")
"""),

        md("## Extreme-fear timing rule vs buy-and-hold (gross and net of costs)"),
        code("""\
if HAVE_REAL:
    s_strat = st.summarize(tr["strat"])
    s_net   = st.summarize(tr["strat_net"])
    s_bh    = st.summarize(tr["bh"])
    n_switch = int(tr["in_mkt"].diff().abs().sum())
    print(f"Active months: {len(tr)}   in-market fraction: {tr['in_mkt'].mean():.1%}   switches: {n_switch}")
    print()
    print(f"STRAT gross : {s_strat['mean']*1200:+.2f}%/yr  HAC t={s_strat['tstat']:+.2f}  SR(ann)={s_strat['sharpe']*12**0.5:+.2f}  hit={s_strat['hit_rate']:.3f}  maxDD={s_strat['max_drawdown']:.1%}")
    print(f"STRAT net   : {s_net['mean']*1200:+.2f}%/yr  HAC t={s_net['tstat']:+.2f}  (5 bps one-way on each switch)")
    print(f"BUY & HOLD  : {s_bh['mean']*1200:+.2f}%/yr  HAC t={s_bh['tstat']:+.2f}  SR(ann)={s_bh['sharpe']*12**0.5:+.2f}  maxDD={s_bh['max_drawdown']:.1%}")
else:
    print(f"Frozen STRAT gross: {R['strat_ann']:+.2f}%/yr  t={R['strat_t']:+.2f}  SR={R['strat_sr']:+.2f}  maxDD={R['strat_dd']:.1%}")
    print(f"Frozen STRAT net  : {R['strat_net_ann']:+.2f}%/yr  t={R['strat_net_t']:+.2f}")
    print(f"Frozen BUY & HOLD : {R['bh_ann']:+.2f}%/yr  t={R['bh_t']:+.2f}  SR={R['bh_sr']:+.2f}  maxDD={R['bh_dd']:.1%}")
print()
print("The timing rule has a LOWER Sharpe than buy-and-hold and gives up ~7.5%/yr")
print("of return for a modestly smaller drawdown -- a poor trade, not an edge.")
"""),

        md("## Conditional next-month return and the Welch test"),
        code("""\
if HAVE_REAL:
    s = sig.iloc[36:]
    fwd = panel["fwd_ret"].iloc[36:]
    fear = fwd[s == 1]; calm = fwd[s == 0]
    from scipy import stats as sc
    wt, wp = sc.ttest_ind(fear, calm, equal_var=False)
    print(f"After extreme fear: {fear.mean()*100:+.3f}%/mo  HAC t={st.summarize(fear)['tstat']:+.2f}  n={len(fear)}")
    print(f"After calm        : {calm.mean()*100:+.3f}%/mo  HAC t={st.summarize(calm)['tstat']:+.2f}  n={len(calm)}")
    print(f"Difference        : {(fear.mean()-calm.mean())*100:+.3f}%/mo  Welch t={wt:+.2f}  p={wp:.3f}")
else:
    print(f"After extreme fear: {R['fear_mean']:+.3f}%/mo  t={R['fear_t']:+.2f}  n={R['fear_n']}")
    print(f"After calm        : {R['calm_mean']:+.3f}%/mo  t={R['calm_t']:+.2f}  n={R['calm_n']}")
    print(f"Difference        : {R['fear_minus_calm']:+.3f}%/mo  Welch t={R['welch_t']:+.2f}  p={R['welch_p']:.3f}")
print()
print("The conditional difference is the WRONG sign and not remotely significant.")
"""),

        md("## Random-timing null: is the fear-timed selection special?"),
        code("""\
if HAVE_REAL:
    frac = tr["in_mkt"].mean()
    rand = st.random_timing_returns(panel, in_mkt_frac=frac, n_draws=2000, seed=261)
    fwd = panel["fwd_ret"].iloc[36:]
    fear_mean = fwd[sig.iloc[36:] == 1].mean()
    perm_p = float((rand >= fear_mean).mean())
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(rand.values*100, bins=40, color=GREY, alpha=0.8)
    ax.axvline(fear_mean*100, color=RED, lw=2, label=f"fear-timed mean = {fear_mean*100:+.2f}%/mo")
    ax.set_title(f"Random in-market timing null (perm p = {perm_p:.2f})")
    ax.set_xlabel("Mean in-market monthly return (%)")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()
    print(f"Permutation p (random >= fear-timed): {perm_p:.3f}")
else:
    print(f"Frozen: random in-market mean = {R['rand_mean']:+.2f}%/mo")
    print(f"Permutation p (random >= fear-timed): {R['perm_p']:.3f}")
print("-> A random pick of the same number of in-market months beats the fear-timed")
print(f"   selection {R['perm_p']*100:.0f}% of the time. The signal carries no timing skill.")
"""),

        md("## Sub-period stability"),
        code("""\
if HAVE_REAL:
    rows = []
    for lab, a, b in [("2006-2012", "2006", "2012"), ("2013-2019", "2013", "2019"), ("2020-2025", "2020", "2025")]:
        sub = tr["strat"][a:b]; sbh = tr["bh"][a:b]
        s = st.summarize(sub); sb = st.summarize(sbh)
        rows.append({"period": lab, "strat_ann": s["mean"]*1200, "t": s["tstat"], "n": s["n"], "bh_ann": sb["mean"]*1200})
        print(f"  {lab}: strat {s['mean']*1200:+.1f}%/yr t={s['tstat']:+.2f} n={s['n']} | buy-hold {sb['mean']*1200:+.1f}%/yr")
    sub_df = pd.DataFrame(rows)
else:
    sub_df = pd.DataFrame([
        {"period": R["sub1_lab"], "strat_ann": R["sub1"], "t": R["sub1_t"], "n": R["sub1_n"], "bh_ann": R["sub1_bh"]},
        {"period": R["sub2_lab"], "strat_ann": R["sub2"], "t": R["sub2_t"], "n": R["sub2_n"], "bh_ann": R["sub2_bh"]},
        {"period": R["sub3_lab"], "strat_ann": R["sub3"], "t": R["sub3_t"], "n": R["sub3_n"], "bh_ann": R["sub3_bh"]},
    ])
    for _, r in sub_df.iterrows():
        print(f"  {r['period']}: strat {r['strat_ann']:+.1f}%/yr t={r['t']:+.2f} n={int(r['n'])} | buy-hold {r['bh_ann']:+.1f}%/yr")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(sub_df))
ax.bar(x-0.2, sub_df["strat_ann"], width=0.4, color=RED, label="extreme-fear timing")
ax.bar(x+0.2, sub_df["bh_ann"], width=0.4, color=GREY, label="buy & hold")
ax.set_xticks(x); ax.set_xticklabels(sub_df["period"])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Annualised return (%/yr)")
ax.set_title("Sub-period: timing rule lags buy-and-hold in every era")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("In every sub-period the rule underperforms simply owning the index.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 261 -- Put-Call-Ratio ===")
print()
print(f"Signal: NONE")
print(f"  Predictive regression slope HAC t = {R['reg_t']:+.2f} (flat).")
print(f"  Extreme-fear months: {R['fear_mean']:+.2f}%/mo vs calm {R['calm_mean']:+.2f}%/mo (Welch t={R['welch_t']:+.2f}, wrong sign).")
print(f"  Random-timing permutation p = {R['perm_p']:.2f} -- no timing skill beyond chance.")
print()
print(f"Tradability: MIRAGE")
print(f"  Rule earns +{R['strat_ann']:.1f}%/yr vs index +{R['bh_ann']:.1f}%/yr; lower Sharpe ({R['strat_sr']:.2f} vs {R['bh_sr']:.2f}).")
print(f"  In-market only {R['in_mkt_frac']*100:.0f}% of months -> misses the bull market.")
print(f"  Net of 5bps switch costs, still +{R['strat_net_ann']:.1f}%/yr (costs are not the problem; the signal is).")
print()
print("Honesty: one-month execution lag built in; price-only S&P (dividends excluded,")
print("  so buy-and-hold's true edge is even larger); ~23yr sample dominated by a few")
print("  crisis spikes -- the few 'extreme' episodes are not independent.")
print()
print("Bottom line: None/Mirage -- the put/call ratio measures fear well but does")
print("  NOT time market bottoms.")
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
    import subprocess
    import sys
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
