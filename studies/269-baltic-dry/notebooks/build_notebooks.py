"""Generate the two narrative notebooks for Study 269 (Baltic-Dry).

    python notebooks/build_notebooks.py

Both notebooks follow the desk beats. The synthetic figures run anywhere,
offline and deterministic; the real-tape cells use the cached/curated monthly
(BDI, S&P) frame if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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
    n_months=493, fp="754cb2059bb5",
    reg_beta=0.0310, reg_t=2.82, reg_r2=0.0452,
    reg_ex2008_beta=0.0106, reg_ex2008_t=1.35, reg_ex2008_n=469,
    sub1_beta=-0.0591, sub1_t=-1.41, sub1_n=177,
    sub2_beta=0.0541, sub2_t=5.08, sub2_n=120,
    sub3_beta=0.0190, sub3_t=2.28, sub3_n=196,
    bh_ann=9.86, bh_t=4.38, bh_sr=0.80, bh_dd=-50.8,
    tim_ann=6.60, tim_t=3.97, tim_sr=0.76, tim_dd=-43.7,
    tim_gross_ann=6.62,
    exc_ann=-3.26, exc_t=-1.90,
    hit=0.5456, base_rate=0.6471,
    n_switch=21, cost_bps=0.21,
    # synthetic positive control
    syn0_beta=-0.008, syn0_t=-1.07,
    syn3_beta=0.086, syn3_t=10.55,
    syn6_beta=0.182, syn6_t=17.50,
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

from baltic_dry import data, strategy as st

# Real tape is built offline from the curated BDI x Shiller S&P (no network).
try:
    df = data.fetch_real(fetch=False)
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    HAVE_REAL = len(df) > 100
except Exception as e:
    df = sig = fwd = None
    HAVE_REAL = False
    print("No real tape:", e)

if HAVE_REAL:
    print(f"Real tape: {df.shape[0]} months  {df.index[0].date()} -> {df.index[-1].date()}")
else:
    print("Real tape unavailable -- frozen headline numbers from R dict will be used")
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
# Study 269 -- Baltic-Dry
## For the curious: does the cost of shipping iron ore tell you where stocks are headed?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

The **Baltic Dry Index (BDI)** measures what it costs to ship dry bulk commodities --
iron ore, coal, grain -- across the world's oceans. Because that freight is booked to
move *physical* stuff, and because the global fleet is fixed in the short run, the BDI
is often sold as a "pure", un-gameable thermometer of real global demand.

The folk wisdom: when freight rates are rising, the world economy is heating up, so the
stock market should follow. **Does the Baltic Dry shipping index lead the stock market?**
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The picture: freight rates and the S&P, side by side

> 💡 *Note on the data:* the official BDI is a licensed Baltic Exchange product with no
clean free feed, so the real tape here pairs a **curated monthly reconstruction** of the
index (anchored to widely reported public levels, log-interpolated between them) with
monthly S&P 500 closes. It reproduces the index's shape and order of magnitude -- the
2008 super-spike, the GFC crash, the 2016 trough, the 2021 rebound -- but it is not the
daily fixing. Treat the numbers as indicative.
"""),
        code("""\
fig, ax1 = plt.subplots(figsize=(11, 5))
if HAVE_REAL:
    ax1.plot(df.index, df["bdi"], color="#1f6feb", lw=1.3, label="Baltic Dry Index (curated)")
    ax1.set_ylabel("BDI level", color="#1f6feb")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["sp500"], color=GREY, lw=1.3, label="S&P 500")
    ax2.set_ylabel("S&P 500 level", color=GREY)
    ax2.set_yscale("log")
    ax2.grid(False)
    ax1.set_title("Baltic Dry Index vs S&P 500 (log scale) -- note the shared 2008 crash")
    # mark the 2008 peak->crash
    ax1.axvspan(pd.Timestamp("2008-05-31"), pd.Timestamp("2009-03-31"), color=RED, alpha=0.12)
else:
    ax1.text(0.5, 0.5, "Real tape unavailable (frozen numbers used below)",
             ha="center", va="center", transform=ax1.transAxes, fontsize=12, color=GREY)
plt.tight_layout()
plt.savefig("../docs/levels.png", dpi=120, bbox_inches="tight")
plt.show()
print("The shaded band is the 2008-09 crisis -- keep an eye on it, it does almost all the work.")
"""),

        md("""\
## Does a rising BDI predict a rising market next month?

The clean test: at each month-end, measure how much the BDI has moved over the last
3 months, then see whether that predicts the S&P's return *next* month (a full month
of lag, so it is genuinely a forecast, not a coincidence).
"""),
        code("""\
if HAVE_REAL:
    reg = st.predictive_regression(sig, fwd)
    print(f"Next-month S&P return on 3M BDI momentum:")
    print(f"  slope = {reg['beta']:+.4f}   HAC t = {reg['tstat']:+.2f}   R^2 = {reg['r2']:.3f}   n = {reg['n']}")
else:
    print(f"Frozen: slope = {R['reg_beta']:+.4f}   HAC t = {R['reg_t']:+.2f}   R^2 = {R['reg_r2']:.3f}")
print()
print(f"A HAC t of +{R['reg_t']:.2f} clears the usual 't >= 2' bar... but hold on.")
"""),

        md("""\
## The catch: it is a 2008 mirage

The whole headline rests on one episode. In late 2008, freight rates **and** the stock
market collapsed *together* -- the BDI fell ~95% and the S&P halved. That single shared
crash is what makes the regression look predictive. Strip the 2008-09 crisis out and the
relationship evaporates.
"""),
        code("""\
if HAVE_REAL:
    mask = ~((df.index.year >= 2008) & (df.index.year <= 2009))
    rex = st.predictive_regression(sig[mask], fwd[mask])
    full_t = st.predictive_regression(sig, fwd)["tstat"]
    print(f"Full sample   : HAC t = {full_t:+.2f}")
    print(f"Ex-2008/09    : HAC t = {rex['tstat']:+.2f}   <- drops below the bar")
else:
    print(f"Full sample   : HAC t = {R['reg_t']:+.2f}")
    print(f"Ex-2008/09    : HAC t = {R['reg_ex2008_t']:+.2f}   <- drops below the bar")
print()
print("Remove one crisis and the 'leading indicator' stops leading.")
"""),

        md("""\
## And you cannot trade it anyway

Suppose you tried the obvious strategy: **stay long the S&P when freight is rising, sit
in cash when it is falling.** Does dodging the down-freight months help?
"""),
        code("""\
if HAVE_REAL:
    tim = st.timing_returns(sig, fwd, one_way_bps=5.0)
    s_bh = st.summarize(tim["buy_hold"]); s_n = st.summarize(tim["timing_net"])
    cum_bh = (1 + tim["buy_hold"]).cumprod()
    cum_tm = (1 + tim["timing_net"]).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum_bh.index, cum_bh.values, color=GREY, lw=1.6, label=f"Buy & hold ({s_bh['mean']*12*100:+.1f}%/yr)")
    ax.plot(cum_tm.index, cum_tm.values, color=RED, lw=1.6, label=f"BDI timing overlay ({s_n['mean']*12*100:+.1f}%/yr)")
    ax.set_yscale("log"); ax.set_ylabel("Growth of \\$1 (log)")
    ax.set_title("Long-when-BDI-rising overlay vs simply staying invested")
    ax.legend()
    plt.tight_layout(); plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight"); plt.show()
    print(f"Buy & hold: {s_bh['mean']*12*100:+.2f}%/yr   Timing: {s_n['mean']*12*100:+.2f}%/yr")
else:
    print(f"Buy & hold: {R['bh_ann']:+.2f}%/yr   Timing: {R['tim_ann']:+.2f}%/yr")
print(f"\\nThe overlay GIVES UP {abs(R['exc_ann']):.1f}%/yr vs buy & hold -- it sits out too many good months.")
"""),

        md(f"""\
## The bottom line for the curious

The Baltic Dry Index is a genuinely useful gauge of *shipping* -- but as a stock-market
crystal ball it is folklore:

- The "predictive" *t*-stat (+{R['reg_t']:.2f}) is almost entirely the **2008 co-crash**;
  strip the crisis and it falls to +{R['reg_ex2008_t']:.2f}.
- The relationship is **negative** in the 1980s-90s -- the sign is not even stable.
- Guessing next month's direction from the BDI is right only {R['hit']*100:.0f}% of the
  time -- *worse* than always betting "up" ({R['base_rate']*100:.0f}%).
- The tradable version **loses {abs(R['exc_ann']):.1f}%/yr** to just staying invested.

A **Weak / Mirage** signal: the BDI moves *with* the world economy, but it does not
reliably move *ahead* of the stock market, and you cannot make money trying.
"""),

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
# Study 269 -- Baltic-Dry
## For the quants: HAC predictive regression, crisis-dependence, timing overlay, positive control

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Synthetic positive control: the engine recovers a planted lead-lag

Before trusting any real-tape reading, confirm the regression lights up when a true
BDI->equity relationship exists and reads ~zero when it does not. The synthetic
generator builds next-month equity returns as ``sp_base + beta * bdi_change[t] + noise``.
"""),
        code("""\
print("Planted beta -> estimated slope (HAC t):")
for beta in [0.0, 0.3, 0.6]:
    d, _ = data.synthetic_series(beta=beta, seed=269)
    s = st.bdi_signal(d, lookback=3); f = st.forward_sp_return(d)
    r = st.predictive_regression(s, f)
    print(f"  beta={beta:.1f}: est={r['beta']:+.3f}  HAC t={r['tstat']:+.2f}  R^2={r['r2']:.3f}")
print()
print("Null (beta=0) reads ~zero; planted lead-lag lights up sharply -> the harness is honest.")
"""),

        md("""\
## Real-tape regression and the crisis-dependence test

> 💡 *In plain words:* the full-sample slope clears t = 2, but a robust signal should not
> hinge on a single episode. We re-run the regression with the 2008-09 crisis removed.
"""),
        code("""\
if HAVE_REAL:
    reg = st.predictive_regression(sig, fwd)
    mask = ~((df.index.year >= 2008) & (df.index.year <= 2009))
    rex = st.predictive_regression(sig[mask], fwd[mask])
    print("=== Next-month S&P return on 3M BDI momentum ===")
    print(f"Full 1985-2026 : beta={reg['beta']:+.4f}  HAC t={reg['tstat']:+.2f}  R^2={reg['r2']:.4f}  n={reg['n']}")
    print(f"Ex-2008/09     : beta={rex['beta']:+.4f}  HAC t={rex['tstat']:+.2f}  n={rex['n']}")
else:
    print("=== Next-month S&P return on 3M BDI momentum (frozen) ===")
    print(f"Full 1985-2026 : beta={R['reg_beta']:+.4f}  HAC t={R['reg_t']:+.2f}  R^2={R['reg_r2']:.4f}  n={R['n_months']}")
    print(f"Ex-2008/09     : beta={R['reg_ex2008_beta']:+.4f}  HAC t={R['reg_ex2008_t']:+.2f}  n={R['reg_ex2008_n']}")
print(f"\\n-> The slope's significance is crisis-dependent: t {R['reg_t']:.2f} -> {R['reg_ex2008_t']:.2f} without 2008/09.")
"""),

        md("## Sub-period slopes: the sign is not even stable"),
        code("""\
periods = [("1985-1999","1985","1999","sub1"), ("2000-2009","2000","2009","sub2"), ("2010-2026","2010","2026","sub3")]
rows = []
for label, a, b, key in periods:
    if HAVE_REAL:
        r = st.predictive_regression(sig[a:b], fwd[a:b])
        beta, t, n = r["beta"], r["tstat"], r["n"]
    else:
        beta, t, n = R[f"{key}_beta"], R[f"{key}_t"], R[f"{key}_n"]
    rows.append({"period": label, "beta": beta, "t": t, "n": n})
    print(f"  {label}: beta={beta:+.4f}  HAC t={t:+.2f}  n={n}")

sub = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9, 4))
colors = [GREEN if t >= 2 else (RED if t < 0 else AMBER) for t in sub["t"]]
ax.bar(sub["period"], sub["t"], color=colors)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t=2 inference bar")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Sub-period slope HAC t-stat (negative in 1985-1999)")
ax.set_ylabel("HAC t-stat"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("Negative in the 1980s-90s, huge in the 2000-2009 supercycle/GFC decade, modest after.")
"""),

        md("## Timing overlay: gross/net vs buy-and-hold (price-only, one-way costs)"),
        code("""\
if HAVE_REAL:
    tim = st.timing_returns(sig, fwd, one_way_bps=5.0)
    s_bh = st.summarize(tim["buy_hold"]); s_g = st.summarize(tim["timing_gross"]); s_n = st.summarize(tim["timing_net"])
    exc = (tim["timing_net"] - tim["buy_hold"]).dropna(); s_e = st.summarize(exc)
    n_switch = int((tim["position"].diff().abs() > 0).sum())
    print("=== Long-when-BDI-rising overlay vs buy-and-hold (price-only) ===")
    print(f"BUY-HOLD     : {s_bh['mean']*12*100:+.2f}%/yr  SR(ann)={s_bh['sharpe']*12**0.5:+.2f}  HAC t={s_bh['tstat']:+.2f}  maxDD={s_bh['max_drawdown']*100:.1f}%")
    print(f"TIMING gross : {s_g['mean']*12*100:+.2f}%/yr")
    print(f"TIMING net   : {s_n['mean']*12*100:+.2f}%/yr  SR(ann)={s_n['sharpe']*12**0.5:+.2f}  HAC t={s_n['tstat']:+.2f}  maxDD={s_n['max_drawdown']*100:.1f}%")
    print(f"TIMING - BH  : {s_e['mean']*12*100:+.2f}%/yr  HAC t={s_e['tstat']:+.2f}  (negative = overlay LOSES)")
    print(f"switches: {n_switch}  avg cost: {tim['cost'].mean()*10000:.2f} bps/mo (turnover is trivial -- costs are NOT the problem)")
else:
    print("=== Long-when-BDI-rising overlay vs buy-and-hold (frozen) ===")
    print(f"BUY-HOLD     : {R['bh_ann']:+.2f}%/yr  SR(ann)={R['bh_sr']:+.2f}  HAC t={R['bh_t']:+.2f}  maxDD={R['bh_dd']:.1f}%")
    print(f"TIMING net   : {R['tim_ann']:+.2f}%/yr  SR(ann)={R['tim_sr']:+.2f}  HAC t={R['tim_t']:+.2f}  maxDD={R['tim_dd']:.1f}%")
    print(f"TIMING - BH  : {R['exc_ann']:+.2f}%/yr  HAC t={R['exc_t']:+.2f}  (negative = overlay LOSES)")
    print(f"switches: {R['n_switch']}  avg cost: {R['cost_bps']:.2f} bps/mo")
print(f"\\nThe overlay's only crumb of value is a shallower drawdown ({R['tim_dd']:.0f}% vs {R['bh_dd']:.0f}%) --")
print("not worth surrendering a third of the equity premium.")
"""),

        md("## Sign hit-rate: worse than always betting 'up'"),
        code("""\
if HAVE_REAL:
    h = st.sign_hit_rate(sig, fwd)
    hit, base = h["hit"], h["base_rate"]
else:
    hit, base = R["hit"], R["base_rate"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["sign(BDI) hit", "base rate (mkt up)"], [hit*100, base*100], color=[RED, GREY])
ax.axhline(50, color="black", ls="--", lw=0.8)
ax.set_ylabel("% correct"); ax.set_title("Directional hit-rate vs unconditional base rate")
for i, v in enumerate([hit*100, base*100]):
    ax.text(i, v + 0.6, f"{v:.1f}%", ha="center")
plt.tight_layout(); plt.show()
print(f"BDI-conditioned direction is right {hit*100:.1f}% of months vs base rate {base*100:.1f}% -- it makes you WORSE.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 269 -- Baltic-Dry ===")
print()
print(f"Signal: WEAK")
print(f"  Full-sample regression HAC t = +{R['reg_t']:.2f} (clears t>=2 naively)")
print(f"  BUT ex-2008/09 t = +{R['reg_ex2008_t']:.2f} (below bar); slope NEGATIVE in 1985-1999;")
print(f"  sign hit-rate {R['hit']*100:.1f}% < base rate {R['base_rate']*100:.1f}%. Not robust -> WEAK, not REAL.")
print()
print(f"Tradability: MIRAGE")
print(f"  Long-when-BDI-rising overlay LOSES {abs(R['exc_ann']):.1f}%/yr to buy-and-hold (t={R['exc_t']:.2f}),")
print(f"  lower Sharpe ({R['tim_sr']:.2f} vs {R['bh_sr']:.2f}); only a marginally shallower drawdown. No edge.")
print()
print(f"Curated-series bias: NAMED -- BDI is a hindsight-curated monthly reconstruction;")
print(f"  in-sample fit is flattered. Real numbers are indicative upper bounds.")
print()
print("Bottom line: Weak/Mirage -- the BDI co-moves with the macro cycle (esp. in 2008)")
print("  but does not reliably LEAD equities, and the tradable read destroys value.")
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
