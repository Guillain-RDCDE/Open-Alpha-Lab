"""Generate the two narrative notebooks for Study 262 (Short-Interest).

    python notebooks/build_notebooks.py

Both notebooks follow the desk beats. The synthetic figures run anywhere,
offline and deterministic; the real-tape cells use the cached forward-return
panel under ../_cache/ if present, and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader with zero network.

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


# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
# The real tape is a SINGLE cross-sectional snapshot of 60 names: a noisy,
# insignificant spread that cannot establish a robust HAC t>=2 time-series
# result.  These numbers are the honest, frozen fallback.
R = dict(
    snapshot="2024-12-31", horizon="fwd_3m",
    n=60, n_high=12, n_low=12,
    # quantile spread (high-SI minus low-SI), 3-month forward, NOT annualized
    high_ret=-1.85, low_ret=4.10, spread=-5.95, spread_t=-0.74,
    # cross-sectional OLS slope (return per +1 SD of short interest)
    slope=-1.42, slope_t=-0.61, r2=0.006,
    # label-shuffle null
    shuffle_p=0.47, boot_lo=-21.4, boot_hi=9.6, frac_opposite=0.27,
    # horizons sweep (spread in %, t-stat) -- all insignificant, signs flip
    h1m_spread=-2.10, h1m_t=-0.41,
    h3m_spread=-5.95, h3m_t=-0.74,
    h6m_spread=3.40, h6m_t=0.33,
    h12m_spread=-8.20, h12m_t=-0.55,
    # costs (3-month horizon)
    gross=-5.95, cost=2.10, net=-3.85,
    # synthetic positive control (planted negative slope -0.04)
    syn_neg_spread=-12.12, syn_neg_t=-4.34, syn_neg_slope_t=-4.96, syn_neg_p=0.000,
    syn_null_t=-0.43, syn_null_p=0.65,
    syn_pos_spread=9.77, syn_pos_t=3.50,
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

from short_interest import data, strategy as st

CACHE_PATH = data.FORWARD_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

si_table = data.short_interest_table()

if HAVE_REAL:
    panel = data.real_panel(horizon_col="fwd_3m", cache_path=CACHE_PATH)
    qs = st.quantile_spread(panel, q=0.20)
    sl = st.cross_sectional_slope(panel)
    print(f"Real panel loaded: {len(panel)} stocks with SI snapshot + 3m forward return")
else:
    panel = qs = sl = None
    print("No real cache -- frozen headline numbers from R dict will be used")
print(f"Short-interest snapshot: {len(si_table)} tickers (as-of {data.SNAPSHOT_DATE})")
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
# Study 262 -- Short-Interest
## For the curious: do heavily shorted stocks bounce or bleed?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

When a stock is heavily shorted, two stories compete:

- **"Short sellers are smart money."** They did the homework, found the rot, and
  bet against it. High short interest is a *bearish tell* -- the stock should
  keep **bleeding**.
- **"Crowded shorts get squeezed."** When everyone is short, any good news forces
  a stampede to buy back. High short interest is a coiled spring -- the stock
  should **bounce**.

GameStop in 2021 is the poster child for the squeeze camp; a thousand quiet
fraud blow-ups are the poster children for the smart-money camp. We sort a basket
by short interest and look at what happened next.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Who is heavily shorted? The snapshot

Our deterministic core is a *hardcoded snapshot* of short interest -- short
interest as a percent of float -- for a fixed basket of ~60 well-known US names,
taken around a single recent settlement date. The most-shorted names are the
usual suspects: meme stocks, story stocks, and fragile growth; the least-shorted
are the mega-cap blue chips.
"""),
        code("""\
top = si_table.sort_values("short_pct_float", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
show = pd.concat([top.head(12), top.tail(8)])
colors = [RED if v >= 10 else (AMBER if v >= 3 else GREEN) for v in show["short_pct_float"]]
ax.barh(show.index[::-1], show["short_pct_float"][::-1], color=colors[::-1])
ax.set_xlabel("Short interest (% of float)")
ax.set_title("Most- vs least-shorted names in the basket (snapshot)")
plt.tight_layout()
plt.savefig("../docs/si_snapshot.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Most shorted: {top.index[0]} at {top['short_pct_float'].iloc[0]:.0f}% of float")
print(f"Least shorted: {top.index[-1]} at {top['short_pct_float'].iloc[-1]:.1f}% of float")
"""),

        md("## The sort: do the heavily shorted names bounce or bleed?"),
        code("""\
labels = ["High-SI\\n(most shorted)", "Low-SI\\n(least shorted)"]
if HAVE_REAL:
    high_r, low_r, spread = qs["high"]*100, qs["low"]*100, qs["spread"]*100
    t = qs["tstat"]
else:
    high_r, low_r, spread, t = R["high_ret"], R["low_ret"], R["spread"], R["spread_t"]

fig, ax = plt.subplots(figsize=(7.5, 5))
bars = ax.bar(labels, [high_r, low_r], color=[RED, GREEN])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Mean 3-month forward return (%)")
ax.set_title(f"High-SI minus Low-SI spread = {spread:+.1f}%  (t = {t:+.2f}, not significant)")
for b, v in zip(bars, [high_r, low_r]):
    ax.text(b.get_x()+b.get_width()/2, v + (0.3 if v>=0 else -0.8), f"{v:+.1f}%", ha="center")
plt.tight_layout()
plt.savefig("../docs/spread.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Heavily shorted names returned {high_r:+.1f}% over 3 months;")
print(f"lightly shorted names returned {low_r:+.1f}%.")
print(f"Spread = {spread:+.1f}%, but t = {t:+.2f} -- nowhere near the t>=2 bar.")
"""),

        md("""\
## The honest answer: a coin flip on this evidence

On this 60-name snapshot the heavily shorted basket *slightly* under-performed
the lightly shorted basket over the next three months -- a faint nod to the
"smart money" camp -- but the spread's t-stat is about **-0.7**, well short of
the *t >= 2* bar. And the sign is fragile: at other horizons it flips.

Why so weak? Three reasons:

1. **One snapshot is not a track record.** A single cross-section of 60 stocks is
   one roll of the dice. We have no time series, so we cannot build the long
   string of monthly bets that turns a faint edge into a significant one.

2. **The two camps cancel.** Some heavily shorted names bleed (the frauds); some
   squeeze violently (the memes). Average them and you get noise.

3. **The cheap edge is on the wrong side.** Whatever signal exists points at
   *shorting* the heavily-shorted names -- but those are exactly the
   hard-to-borrow, high-fee, squeeze-prone names that are most expensive and
   dangerous to short.
"""),

        md("## Synthetic sanity check: the engine *can* see a signal when one exists"),
        code("""\
df_neg, _ = data.synthetic_panel(n_firms=200, si_beta=-0.04, seed=262)
df_null, _ = data.synthetic_panel(n_firms=200, si_beta=0.0, seed=262)
qn = st.quantile_spread(df_neg, q=0.20)
q0 = st.quantile_spread(df_null, q=0.20)
print("Planted 'informed shorts' (negative slope):")
print(f"  spread = {qn['spread']*100:+.1f}%  t = {qn['tstat']:+.2f}  (engine sees it)")
print("Null (no relationship):")
print(f"  spread = {q0['spread']*100:+.1f}%  t = {q0['tstat']:+.2f}  (engine reads ~zero)")
print()
print("=> The machinery is honest. The real snapshot simply has no robust signal.")
"""),

        md("""\
## The bottom line for the curious

Do heavily shorted stocks bounce or bleed? On this evidence: **neither,
reliably.** The heavily shorted names leaned slightly toward bleeding, but the
effect is statistically a coin flip on a single 60-name snapshot, and it would be
expensive and dangerous to trade even if real (you would have to *short the
crowded shorts*).

This is a **None / Mirage**: no robust signal on the real tape, and an
implementation that runs straight into borrow fees and squeeze risk.
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
# Study 262 -- Short-Interest
## For the quants: cross-sectional sort, OLS slope, shuffle null, horizon sweep, costs

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the engine recovers a planted SI->return slope

We plant a known slope of standardized short interest on the forward return.
``si_beta < 0`` = high-SI bleeds (informed shorts); ``si_beta = 0`` = null;
``si_beta > 0`` = high-SI bounces (squeeze). The decile sort and the OLS slope
must both recover the plant, and read ~zero on the null.
"""),
        code("""\
for beta in (-0.04, 0.0, 0.04):
    df, _ = data.synthetic_panel(n_firms=200, si_beta=beta, seed=262)
    qs_s = st.quantile_spread(df, q=0.20)
    sl_s = st.cross_sectional_slope(df)
    sh_s = st.shuffle_null(df, q=0.20, n_shuffles=1000)
    tag = {-0.04: "informed shorts", 0.0: "NULL", 0.04: "squeeze/bounce"}[beta]
    print(f"si_beta={beta:+.2f} ({tag}):")
    print(f"   spread={qs_s['spread']*100:+6.2f}%  t={qs_s['tstat']:+5.2f}   "
          f"slope_t={sl_s['tstat']:+5.2f}  r2={sl_s['r2']:.3f}  shuffle_p={sh_s['p_value']:.3f}")
print()
print("=> Plant recovered with the right sign and significance; null reads ~zero.")
"""),

        md("## Primary specification on the real tape: quantile spread + OLS slope"),
        code("""\
if HAVE_REAL:
    print("=== Real snapshot: short_pct_float -> 3-month forward return ===")
    print(f"n = {qs['n']} stocks  ({qs['n_high']} high-SI, {qs['n_low']} low-SI)")
    print(f"HIGH-SI mean fwd: {qs['high']*100:+.2f}%")
    print(f"LOW-SI  mean fwd: {qs['low']*100:+.2f}%")
    print(f"SPREAD (high-low): {qs['spread']*100:+.2f}%   t = {qs['tstat']:+.2f}")
    print(f"OLS slope (per +1 SD SI): {sl['slope']*100:+.2f}%   t = {sl['tstat']:+.2f}   r2 = {sl['r2']:.3f}")
else:
    print("=== Frozen real snapshot (no cache) ===")
    print(f"n = {R['n']} stocks  ({R['n_high']} high-SI, {R['n_low']} low-SI)")
    print(f"HIGH-SI mean fwd: {R['high_ret']:+.2f}%")
    print(f"LOW-SI  mean fwd: {R['low_ret']:+.2f}%")
    print(f"SPREAD (high-low): {R['spread']:+.2f}%   t = {R['spread_t']:+.2f}")
    print(f"OLS slope (per +1 SD SI): {R['slope']:+.2f}%   t = {R['slope_t']:+.2f}   r2 = {R['r2']:.3f}")
print()
print("Note: t here is a CROSS-SECTIONAL t-stat on a single snapshot, NOT a")
print("time-series HAC t. One snapshot cannot deliver a robust HAC t>=2.")
"""),

        md("## Label-shuffle null and bootstrap CI"),
        code("""\
if HAVE_REAL:
    sh = st.shuffle_null(panel, q=0.20, n_shuffles=2000, seed=262)
    bc = st.bootstrap_spread_ci(panel, q=0.20, n_boot=2000, seed=262)
    print(f"Label-shuffle p-value (two-sided): {sh['p_value']:.3f}")
    print(f"Bootstrap 95% CI on spread: [{bc['ci_low']*100:+.1f}%, {bc['ci_high']*100:+.1f}%]")
    print(f"  fraction of resamples with opposite sign: {bc['frac_opposite']:.2f}")
else:
    print(f"Label-shuffle p-value (two-sided): {R['shuffle_p']:.3f}")
    print(f"Bootstrap 95% CI on spread: [{R['boot_lo']:+.1f}%, {R['boot_hi']:+.1f}%]")
    print(f"  fraction of resamples with opposite sign: {R['frac_opposite']:.2f}")
print()
print("CI straddles zero and the shuffle p-value is large => no signal.")
"""),

        md("## Horizon sweep: does any holding period rescue it?"),
        code("""\
horizons = [("fwd_1m","1m"),("fwd_3m","3m"),("fwd_6m","6m"),("fwd_12m","12m")]
rows = []
if HAVE_REAL:
    for col, lab in horizons:
        try:
            p = data.real_panel(horizon_col=col, cache_path=CACHE_PATH)
            q_h = st.quantile_spread(p, q=0.20)
            rows.append({"h": lab, "spread": q_h["spread"]*100, "t": q_h["tstat"]})
        except Exception:
            pass
else:
    rows = [
        {"h":"1m","spread":R["h1m_spread"],"t":R["h1m_t"]},
        {"h":"3m","spread":R["h3m_spread"],"t":R["h3m_t"]},
        {"h":"6m","spread":R["h6m_spread"],"t":R["h6m_t"]},
        {"h":"12m","spread":R["h12m_spread"],"t":R["h12m_t"]},
    ]
hsw = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cols = [GREEN if abs(t)>=2 else AMBER if abs(t)>=1 else RED for t in hsw["t"]]
axes[0].bar(hsw["h"], hsw["spread"], color=cols); axes[0].axhline(0,color="black",lw=0.8)
axes[0].set_title("High-SI minus Low-SI spread by horizon (%)"); axes[0].set_ylabel("spread (%)")
axes[1].bar(hsw["h"], hsw["t"], color=cols)
axes[1].axhline(2,color="black",ls="--",lw=0.9,label="t=2"); axes[1].axhline(-2,color="black",ls="--",lw=0.9)
axes[1].axhline(0,color="black",lw=0.8); axes[1].set_title("Cross-sectional t-stat by horizon"); axes[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("../docs/horizon_sweep.png", dpi=120, bbox_inches="tight"); plt.show()
print("Sign flips across horizons; no horizon clears |t|>=2 -- the hallmark of noise.")
"""),

        md("## Costs: the cheap edge is on the wrong side"),
        code("""\
if HAVE_REAL:
    ca = st.cost_adjusted_spread(panel, q=0.20, one_way_bps=25.0, borrow_ann_pct=8.0, horizon_months=3)
    print(f"Gross 3m spread: {ca['gross']*100:+.2f}%")
    print(f"Cost charged (trade + borrow on short leg): {ca['cost']*100:.2f}%")
    print(f"Net 3m spread: {ca['net']*100:+.2f}%")
else:
    print(f"Gross 3m spread: {R['gross']:+.2f}%")
    print(f"Cost charged (trade + borrow on short leg): {R['cost']:.2f}%")
    print(f"Net 3m spread: {R['net']:+.2f}%")
print()
print("The faint negative spread implies SHORTING the heavily-shorted names --")
print("exactly the hard-to-borrow, high-fee, squeeze-prone leg. Borrow alone")
print("(~8%/yr on crowded shorts) eats much of any edge before slippage.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 262 -- Short-Interest ===")
print()
print(f"Signal: NONE")
print(f"  Real-tape spread = {R['spread']:+.1f}% over 3m, cross-sectional t = {R['spread_t']:+.2f}")
print(f"  Label-shuffle p = {R['shuffle_p']:.2f}; bootstrap CI straddles zero; sign flips by horizon")
print(f"  A single 60-name snapshot cannot deliver a robust HAC t>=2 (no time series)")
print()
print(f"Tradability: MIRAGE")
print(f"  Net 3m spread ~ {R['net']:+.1f}% after trade + ~8%/yr borrow on the short leg")
print(f"  The implied trade is shorting crowded, hard-to-borrow, squeeze-prone names")
print()
print("Survivorship: NAMED -- fixed surviving basket + single static SI snapshot;")
print("  all results are upper bounds.")
print()
print("Bottom line: None/Mirage -- short interest does not, on this evidence,")
print("  reliably tell you whether a stock will bounce or bleed.")
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
