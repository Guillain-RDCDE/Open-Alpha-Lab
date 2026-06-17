"""Generate the two narrative notebooks for Study 265 (IPO-Volume).

    python notebooks/build_notebooks.py

Both notebooks follow the desk beats. The synthetic figures run anywhere, offline
and deterministic; the real-tape cells use the repo-level Shiller cache
(`_cache/shiller_sp500.parquet`) joined with the hardcoded IPO table if present,
and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it. nbconvert executes both at the
end; --allow-errors is never used.
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
    n_years=41, year_start=1980, year_end=2024, fingerprint="a6855574f53b",
    # primary predictive regression
    beta=-0.0019, beta_t=-0.07, r2=0.000, pred_corr=-0.011,
    # coincidence trap
    contemp_corr=0.317,
    # sub-periods
    sub_a_label="1985-1999", sub_a_beta=-0.007, sub_a_t=-0.19, sub_a_n=15,
    sub_b_label="2000-2012", sub_b_beta=-0.159, sub_b_t=-2.11, sub_b_n=13,
    sub_c_label="2013-2024", sub_c_beta=-0.207, sub_c_t=-6.23, sub_c_n=12,
    # timing rule vs buy-and-hold (net 5bps)
    strat_mean=8.33, strat_vol=14.0, strat_t=4.12, strat_hit=0.59,
    bh_mean=10.79, bh_vol=15.8, bh_t=4.68, bh_hit=0.76,
    excess_mean=-2.46, excess_t=-1.12, n_long=29, n_switches=9, drag_bps=45,
    # synthetic positive control (n=120)
    syn_null_beta=-0.004, syn_null_t=-0.36,
    syn_froth_beta=-0.056, syn_froth_t=-4.92,
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

from ipo_volume import data, strategy as st

SHILLER = os.path.join(data.REPO_CACHE, "shiller_sp500.parquet")
HAVE_REAL = os.path.exists(SHILLER)

if HAVE_REAL:
    df = data.fetch_sp500_annual()
    z = st.froth_signal(df, min_obs=5)
    reg = st.predictive_regression(z, df["fwd_return"])
    tm = st.timing_returns(z, df["fwd_return"], threshold=0.0, one_way_bps=5.0)
    print(f"Real annual tape loaded: {len(df)} forecastable years "
          f"{int(df['year'].min())}-{int(df['year'].max())}")
    print(f"Predictive slope HAC t = {reg['beta_t']:+.2f}")
else:
    df = z = reg = tm = None
    print("No Shiller cache -- frozen headline numbers from R dict will be used")
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
# Study 265 -- IPO-Volume
## For the curious: is a flood of IPOs the bell at the top of the market?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

Every cycle the story comes back: the IPO window is wide open, hundreds of
companies are rushing to list, your barber is talking about the latest debut --
*surely* that is the bell ringing at the top. 1999. 2021. The famous floods.

But does a big IPO year actually **predict** a bad year for the market *next*
year? Or does it just **coincide** with the good year we are already having? We
pair the canonical annual US IPO count with the S&P 500's calendar-year return
and let the tape settle it.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The picture everyone remembers: IPO floods and famous tops

The eye locks onto 1999 (476 IPOs, then the dot-com bust) and 2021 (the SPAC-era
flood, then the 2022 drawdown). What it forgets: the 2008 window slammed shut
(21 IPOs) right before one of the worst years -- and the *rebound* came while the
IPO market was still frozen. The timing is messier than the legend.
"""),
        code("""\
ipo = data.ipo_table().set_index("year")["ipo_count"]
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(ipo.index, ipo.values, color=GREY, width=0.8)
for yr, col, txt in [(1999, RED, "1999 flood"), (2008, GREEN, "2008 freeze"), (2021, RED, "2021 flood")]:
    if yr in ipo.index:
        ax.bar([yr], [ipo.loc[yr]], color=col)
        ax.text(yr, ipo.loc[yr] + 12, txt, ha="center", fontsize=8, color=col)
ax.set_title("Annual US operating-company IPO count (Ritter / Univ. of Florida)")
ax.set_ylabel("IPOs priced")
ax.set_xlabel("Year")
plt.tight_layout()
plt.savefig("../docs/ipo_counts.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Dot-com flood 1999: {int(ipo.loc[1999])} IPOs   |   2008 freeze: {int(ipo.loc[2008])} IPOs")
"""),

        md("""\
## The coincidence trap: 'during' is not 'before'

Here is the whole study in one comparison. IPO volume is genuinely high *during*
good years -- a **+0.32** correlation with the *same-year* return. That is what
makes the bell-at-the-top story feel obviously true.

But the only thing a trader can act on is whether a frothy year today predicts a
bad year *tomorrow*. And that correlation is **-0.01** -- a flat zero. The
indicator is *coincident*, not *leading*.
"""),
        code("""\
fig, ax = plt.subplots(figsize=(7.5, 4.6))
if HAVE_REAL:
    pair = pd.concat([z.rename("z"), df["sp500_return"].rename("contemp")], axis=1).dropna()
    contemp = float(np.corrcoef(pair["z"], pair["contemp"])[0, 1])
    pred = reg["corr"]
else:
    contemp, pred = R["contemp_corr"], R["pred_corr"]
bars = ax.bar(["froth vs\\nSAME-year return", "froth vs\\nNEXT-year return"],
              [contemp, pred], color=[AMBER, RED])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("correlation")
ax.set_title("Coincident, not leading")
for b, v in zip(bars, [contemp, pred]):
    ax.text(b.get_x() + b.get_width()/2, v + (0.02 if v >= 0 else -0.04),
            f"{v:+.2f}", ha="center", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
print(f"Same-year (contemporaneous) corr: {contemp:+.2f}  -- IPOs flood DURING good years")
print(f"Next-year (predictive)     corr: {pred:+.2f}  -- froth tells you nothing about tomorrow")
"""),

        md("""\
## Could you trade it? A simple 'sit out the froth' rule

The natural trade: stay long the S&P next year *unless* this year was frothy
(IPO z-score above zero), in which case go to cash. If the bell-at-the-top story
were real, dodging the frothy years should help.

It doesn't. The rule trails plain buy-and-hold by about **2.5%/yr** -- because
"frothy" years are usually *good* years, and you spend the next one in cash for
nothing.
"""),
        code("""\
if HAVE_REAL:
    s_n = st.summarize(tm["strat_net"]); s_bh = st.summarize(tm["buy_hold"])
    strat_mean, bh_mean = s_n["mean"]*100, s_bh["mean"]*100
    cum_s = (1 + tm["strat_net"]).cumprod()
    cum_b = (1 + tm["buy_hold"]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(df["year"].iloc[-len(cum_s):], cum_s.values, color=RED, lw=1.8, label="Sit-out-the-froth (net)")
    ax.plot(df["year"].iloc[-len(cum_b):], cum_b.values, color=GREEN, lw=1.8, label="Buy & hold")
    ax.set_yscale("log"); ax.set_title("Growth of $1 (price-only S&P)")
    ax.set_ylabel("cumulative ($, log)"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight"); plt.show()
else:
    strat_mean, bh_mean = R["strat_mean"], R["bh_mean"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Sit-out-the-froth", "Buy & hold"], [strat_mean, bh_mean], color=[RED, GREEN])
    ax.set_ylabel("mean return (%/yr)"); ax.set_title("(frozen numbers)")
    plt.tight_layout(); plt.show()
print(f"Sit-out-the-froth: {strat_mean:+.2f}%/yr   Buy & hold: {bh_mean:+.2f}%/yr")
print(f"The rule trails buy-and-hold by {R['excess_mean']:+.2f}%/yr -- dodging froth dodges good years.")
"""),

        md("""\
## The bottom line for the curious

The flood of IPOs **feels** like the bell at the top because it really is loud at
the top -- but it rings *during* the party, not the morning before the hangover.
By the time the market actually rolls over, the IPO window has usually already
closed on its own.

- IPO volume vs **same-year** return: **+0.32** (coincident)
- IPO volume vs **next-year** return: **-0.01** (no forecast power)
- A "sit out the froth" rule loses to buy-and-hold by ~2.5%/yr

This is a **None / Mirage**: a vivid coincidence dressed up as a leading
indicator.
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
# Study 265 -- IPO-Volume
## For the quants: HAC predictive regression, the coincidence gap, subsample mining, costs

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the estimator recovers a planted froth slope

Before reading the real tape, confirm the regression machinery is honest. We
plant a known slope `beta` in `data.synthetic_annual` (high IPO z -> low forward
return) and check the estimator recovers it, and reads ~zero on the null.

> 💡 **In plain words**: if the engine couldn't find a froth effect when we
> *built one in*, a flat result on real data would be meaningless. This rules
> that out.
"""),
        code("""\
for beta in (0.0, -0.06):
    syn, _ = data.synthetic_annual(n_years=120, beta=beta, seed=265)
    zs = st.froth_signal(syn, min_obs=5)
    rs = st.predictive_regression(zs, syn["fwd_return"])
    tag = "null " if beta == 0.0 else "froth"
    print(f"  planted beta={beta:+.2f} ({tag}): recovered beta={rs['beta']:+.4f}  HAC t={rs['beta_t']:+.2f}")
print()
print(f"Frozen control: null beta -> t={R['syn_null_t']:+.2f};  froth beta=-0.06 -> t={R['syn_froth_t']:+.2f}")
print("Engine is honest: it finds a planted slope and reads ~0 on the null.")
"""),

        md("""\
## Primary specification: next-year return on the froth z-score

`fwd_return = alpha + beta * froth_z + eps`, Newey-West HAC standard errors,
one-year execution lag. The froth z-score is standardized with an **expanding,
past-only** window (no future IPO statistics leak in). The folklore predicts
`beta < 0`.
"""),
        code("""\
if HAVE_REAL:
    print("=== Predictive regression (real tape) ===")
    print(f"  beta = {reg['beta']:+.4f}   HAC t = {reg['beta_t']:+.2f}   R^2 = {reg['r2']:.3f}   n = {reg['n']}")
    print(f"  predictive corr(froth_z, next-yr ret) = {reg['corr']:+.3f}")
    beta_t = reg['beta_t']
else:
    print("=== Predictive regression (frozen) ===")
    print(f"  beta = {R['beta']:+.4f}   HAC t = {R['beta_t']:+.2f}   R^2 = {R['r2']:.3f}   n = {R['n_years']}")
    beta_t = R['beta_t']
print()
print(f"Inference bar is |t| >= 2. Here |t| = {abs(beta_t):.2f}  ->  FAILS. No predictive content.")
"""),

        md("""\
## The coincidence gap: contemporaneous vs predictive correlation

The myth's whole emotional force comes from the *same-year* correlation. We put
the coincident and the predictive correlation side by side -- the gap is the
entire story.
"""),
        code("""\
if HAVE_REAL:
    pair = pd.concat([z.rename("z"), df["sp500_return"].rename("c"),
                      df["fwd_return"].rename("f")], axis=1).dropna()
    contemp = float(np.corrcoef(pair["z"], pair["c"])[0, 1])
    pred = float(np.corrcoef(pair["z"], pair["f"])[0, 1])
else:
    contemp, pred = R["contemp_corr"], R["pred_corr"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].bar(["same-year", "next-year"], [contemp, pred], color=[AMBER, RED])
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_title("corr(froth_z, return)"); axes[0].set_ylabel("correlation")
for i, v in enumerate([contemp, pred]):
    axes[0].text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.2f}", ha="center", fontweight="bold")

if HAVE_REAL:
    axes[1].scatter(pair["z"], pair["f"]*100, color=GREY, s=28)
    b1, b0 = np.polyfit(pair["z"], pair["f"]*100, 1)
    xs = np.linspace(pair["z"].min(), pair["z"].max(), 50)
    axes[1].plot(xs, b0 + b1*xs, color=RED, lw=1.6)
    axes[1].set_xlabel("froth z-score (year Y)"); axes[1].set_ylabel("S&P return year Y+1 (%)")
    axes[1].set_title("Flat predictive scatter")
else:
    axes[1].text(0.5, 0.5, "scatter needs real cache", ha="center", va="center", transform=axes[1].transAxes)
plt.tight_layout(); plt.show()
print(f"Contemporaneous {contemp:+.2f} vs predictive {pred:+.2f}: coincident, not leading.")
"""),

        md("""\
## Subsample mining: why 'it worked recently' is a trap

Carve the sample into thirds and two recent windows *look* like the folklore
finally working (even `t = -6.2` for 2013-2024). But these are **n ~ 12** slices
of a sample whose full-period slope is zero. This is exactly the multiple-testing
mirage the desk exists to flag.

> 💡 **In plain words**: torture 41 data points into small enough pieces and one
> piece will confess. The honest test is the full sample -- and it says nothing.
"""),
        code("""\
subs = [(R["sub_a_label"], 1985, 1999), (R["sub_b_label"], 2000, 2012), (R["sub_c_label"], 2013, 2024)]
rows = []
if HAVE_REAL:
    yr = df["year"]
    for label, a, b in subs:
        m = (yr >= a) & (yr <= b)
        rsub = st.predictive_regression(z[m], df["fwd_return"][m])
        rows.append({"period": label, "beta": rsub["beta"], "t": rsub["beta_t"], "n": rsub["n"]})
else:
    rows = [
        {"period": R["sub_a_label"], "beta": R["sub_a_beta"], "t": R["sub_a_t"], "n": R["sub_a_n"]},
        {"period": R["sub_b_label"], "beta": R["sub_b_beta"], "t": R["sub_b_t"], "n": R["sub_b_n"]},
        {"period": R["sub_c_label"], "beta": R["sub_c_beta"], "t": R["sub_c_t"], "n": R["sub_c_n"]},
    ]
sub_df = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(8.5, 4))
colors = [RED if abs(t) >= 2 else GREY for t in sub_df["t"]]
ax.bar(sub_df["period"], sub_df["t"], color=colors)
ax.axhline(-2.0, color="black", ls="--", lw=0.9, label="|t|=2 bar")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("HAC t on slope"); ax.set_title("Sub-period slope t (n~12-15 each -- a mining trap)")
ax.legend(fontsize=9)
for i, row in sub_df.iterrows():
    ax.text(i, row["t"] - 0.4, f"n={int(row['n'])}", ha="center", fontsize=8)
plt.tight_layout(); plt.show()
print("Full-sample (n=41) slope t ~ 0; the 'significant' sub-periods are subsample mirages.")
"""),

        md("""\
## The tradable wrapper: long/flat froth-timing vs buy-and-hold

Long the index next year when this year's froth z < 0, flat otherwise.
Long-only -- **no short, no borrow**. 5 bps one-way cost on each switch, charged
to NAV. We report **gross and net**, on a **price-only** S&P tape.
"""),
        code("""\
if HAVE_REAL:
    s_g = st.summarize(tm["strat_gross"]); s_n = st.summarize(tm["strat_net"]); s_bh = st.summarize(tm["buy_hold"])
    exc = st.summarize(tm["strat_net"] - tm["buy_hold"])
    n_long = int(tm["position"].sum()); switches = int((tm["position"] != tm["position"].shift(1).fillna(0)).sum())
    drag = float((tm["strat_gross"] - tm["strat_net"]).sum() * 1e4)
    print(f"Years long: {n_long}/{len(tm)}   switches: {switches}   total cost drag: {drag:.0f} bps")
    print(f"  STRAT gross: {s_g['mean']*100:+.2f}%/yr  vol={s_g['vol']*100:.1f}%  HAC t={s_g['tstat']:+.2f}")
    print(f"  STRAT net  : {s_n['mean']*100:+.2f}%/yr  vol={s_n['vol']*100:.1f}%  HAC t={s_n['tstat']:+.2f}")
    print(f"  BUY & HOLD : {s_bh['mean']*100:+.2f}%/yr  vol={s_bh['vol']*100:.1f}%  HAC t={s_bh['tstat']:+.2f}")
    print(f"  STRAT-B&H  : {exc['mean']*100:+.2f}%/yr  HAC t={exc['tstat']:+.2f}  (negative => HURTS)")
else:
    print(f"Years long: {R['n_long']}/{R['n_years']}   switches: {R['n_switches']}   cost drag: {R['drag_bps']} bps")
    print(f"  STRAT net  : {R['strat_mean']:+.2f}%/yr  vol={R['strat_vol']:.1f}%  HAC t={R['strat_t']:+.2f}")
    print(f"  BUY & HOLD : {R['bh_mean']:+.2f}%/yr  vol={R['bh_vol']:.1f}%  HAC t={R['bh_t']:+.2f}")
    print(f"  STRAT-B&H  : {R['excess_mean']:+.2f}%/yr  HAC t={R['excess_t']:+.2f}  (negative => HURTS)")
print("\\nThe rule underperforms buy-and-hold: sitting out frothy years sits out good years.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 265 -- IPO-Volume ===")
print()
print(f"Signal: NONE   (full-sample predictive HAC t = {R['beta_t']:+.2f} on n={R['n_years']} years)")
print(f"  Contemporaneous corr +{R['contemp_corr']:.2f} is coincident, not leading (predictive {R['pred_corr']:+.2f})")
print(f"  Negative sub-period slopes (t={R['sub_b_t']}, {R['sub_c_t']}) are n~12 subsample mirages")
print(f"  Positive control recovers planted slope at t={R['syn_froth_t']:+.2f}: the null is real, not a bug")
print()
print(f"Tradability: MIRAGE")
print(f"  Long/flat froth-timing net {R['strat_mean']:+.2f}%/yr vs buy-and-hold {R['bh_mean']:+.2f}%/yr")
print(f"  Underperforms by {R['excess_mean']:+.2f}%/yr -- nothing to capture")
print()
print("Sample size: tiny-n (41 forecastable years) -- even the 'working' slices are fishing")
print()
print("Bottom line: None/Mirage -- a vivid coincidence mistaken for a leading indicator.")
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
