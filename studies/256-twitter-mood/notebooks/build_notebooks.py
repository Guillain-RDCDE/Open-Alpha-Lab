"""Generate the two narrative notebooks for Study 256 (Twitter-Mood).

    python notebooks/build_notebooks.py

Both notebooks follow the desk beats. The synthetic figures run anywhere,
offline and deterministic; the real-tape cells use the cached ^GSPC daily
series under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader, online or off.

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
    n_days=207, window="2008-02-28 -> 2008-12-19", fingerprint="a8dce11f0ac1",
    # lead-lag sweep (HAC t)
    beta1=0.00327, t1=0.90, r2_1=0.0043,
    beta2=0.00233, t2=0.63,
    beta3=0.00192, t3=0.52,
    beta4=0.00203, t4=0.56,
    beta5=0.00244, t5=0.68,
    bonf_bar=2.58,
    # shuffle null
    perm_p1=0.35, perm_p2=0.51, perm_p3=0.57,
    # directional
    hit_rate=51.3, uncond_up=50.8, edge_pts=0.5,
    # trading rule
    long_gross_ann=3.9, long_gross_t=0.17, long_net_ann=3.2, long_net_t=0.14,
    ls_gross_ann=50.8, ls_gross_t=1.32, ls_net_ann=48.1, ls_net_t=1.25,
    bh_ann=-43.1, bh_t=-1.18,
    # positive control
    pc_null_beta=-0.00054, pc_null_t=-0.57,
    pc_mid_beta=0.00146, pc_mid_t=1.57,
    pc_hi_beta=0.00346, pc_hi_t=3.71,
)


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

from twitter_mood import data, strategy as st

CACHE_PATH = data.INDEX_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    df_real = data.build_real_panel(fetch=False, cache_path=CACHE_PATH)
    sweep = st.leadlag_sweep(df_real, max_lag=5)
    print(f"Real panel loaded: {len(df_real)} business days "
          f"{df_real.index[0].date()} -> {df_real.index[-1].date()}")
else:
    df_real = sweep = None
    print("No real cache -- frozen headline numbers from R dict will be used")
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
# Study 256 -- Twitter-Mood
## For the curious: can the mood of Twitter tell you where the market goes tomorrow?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

In 2011 a paper went viral: **"Twitter mood predicts the stock market"** (Bollen, Mao
& Zeng). By scoring the *Calm* of ~10 million tweets, the authors claimed they could
forecast the Dow's next-day direction with **86.7% accuracy** -- and an entire
"social-media alpha" industry was born.

We rebuild the cleanest version of that test on the real S&P 500 tape over the exact
2008 window Bollen used, and ask the honest question: **is there a tradable mood ->
market lead-lag, or is the 87% a mirage?**
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The mood tape and the market

We use a curated daily **Calm** index pinned to documented 2008 public-mood moments
(the spring calm, the Lehman-week panic, the post-election + Thanksgiving spike Bollen
himself highlighted). *Note:* the original GPOMS feed was never released, so this is a
**stylized reconstruction**, not the proprietary series -- which is exactly the kind of
caveat that should make you suspicious of the headline claim.
"""),
        code("""\
calm = data.calm_index()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(calm.index, calm.values, color=GREY, lw=1.6)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.fill_between(calm.index, calm.values, 0, where=(calm.values >= 0), color=GREEN, alpha=.25)
ax.fill_between(calm.index, calm.values, 0, where=(calm.values < 0),  color=RED,   alpha=.25)
ax.set_title("Curated daily 'Calm' mood index (2008 window) -- a stylized reconstruction")
ax.set_ylabel("Calm (standardized)")
ax.annotate("Lehman week", xy=(pd.Timestamp("2008-09-15"), calm.min()),
            xytext=(pd.Timestamp("2008-06-20"), -1.3), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))
plt.tight_layout(); plt.savefig("../docs/calm_index.png", dpi=120, bbox_inches="tight"); plt.show()
print(f"Calm index: {len(calm)} business days, range [{calm.min():.2f}, {calm.max():.2f}]")
"""),

        md("""\
## Does yesterday's Calm forecast today's return?

The heart of Bollen's claim: **Calm leads the market**. We regress the next day's
return on today's Calm and look at the slope's *t*-statistic. The desk bar for "real"
is |*t*| >= 2.
"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 4.5))
if HAVE_REAL:
    lags = sweep["lag"].astype(int).tolist()
    tstats = sweep["tstat"].tolist()
else:
    lags = [1, 2, 3, 4, 5]
    tstats = [R["t1"], R["t2"], R["t3"], R["t4"], R["t5"]]
colors = [GREEN if abs(t) >= 2 else RED for t in tstats]
ax.bar([str(L) for L in lags], tstats, color=colors)
ax.axhline(2.0,  color="black", ls="--", lw=0.9, label="t = 2 inference bar")
ax.axhline(-2.0, color="black", ls="--", lw=0.9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Lead-lag HAC t-stat: next-day return on lagged Calm (every lag is noise)")
ax.set_xlabel("Lag (trading days)"); ax.set_ylabel("HAC t-stat")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print(f"Lag-1 slope HAC t = +{R['t1']:.2f}  -- nowhere near the t>=2 bar.")
print(f"Sweeping 5 lags raises the bar to |t| ~ {R['bonf_bar']:.2f} (Bonferroni); nothing clears it.")
"""),

        md("""\
## The base-rate trap: "51% accuracy" is not 87%

A binary "up tomorrow?" rule looks good simply because **the market drifts up most
days**. The honest comparison is against the *unconditional* up-rate, not a coin flip.
"""),
        code("""\
if HAVE_REAL:
    hr = st.directional_hitrate(df_real, lookback=20)
    hit, uncond = hr["hit_rate"]*100, hr["uncond_up_rate"]*100
else:
    hit, uncond = R["hit_rate"], R["uncond_up"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Calm rule", "Unconditional\\nup-rate"], [hit, uncond], color=[AMBER, GREY])
ax.axhline(50, color="black", ls="--", lw=0.8)
ax.set_ylim(40, 60); ax.set_ylabel("Next-day hit-rate (%)")
ax.set_title("Calm rule vs the honest baseline")
for i, v in enumerate([hit, uncond]):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=11)
plt.tight_layout(); plt.show()
print(f"Edge over baseline: +{hit-uncond:.1f} pts. Bollen's '86.7%' does not survive.")
"""),

        md("""\
## "But the long-short rule makes 48%/yr!" -- the 2008 crash trap

A long-short version of the Calm rule *looks* spectacular -- but only because the test
window **is the 2008 crash**. Any rule that ends up net-short into a -43%/yr panic mints
money for a reason that has nothing to do with Twitter. And it *still* fails the *t* >= 2
bar.
"""),
        code("""\
print("Trading rule over the 2008 window (1-day execution lag, costs on NAV):")
print(f"  Long/flat : {R['long_net_ann']:+.1f}%/yr net  HAC t = {R['long_net_t']:+.2f}   (flat)")
print(f"  Long/short: {R['ls_net_ann']:+.1f}%/yr net  HAC t = {R['ls_net_t']:+.2f}   (looks great...)")
print(f"  Buy & hold: {R['bh_ann']:+.1f}%/yr        HAC t = {R['bh_t']:+.2f}   (<- the crash)")
print()
print(f"The +48%/yr is a net-short-into-the-crash artefact, NOT a mood signal,")
print(f"and t = {R['ls_net_t']:.2f} does not clear the inference bar anyway.")
"""),

        md("""\
## The bottom line for the curious

The Twitter-mood story is a textbook **mirage**:

- The lead-lag slope is *t* = 0.90 -- pure noise; a permutation null swallows it (*p* = 0.35).
- The "87% accuracy" collapses to a 51.3% hit-rate, +0.5 pts over the base rate.
- The only profitable trade is a 2008-crash artefact that doesn't clear *t* >= 2.
- And the window is only ~10 trading months (n = 207) -- far too small to detect a
  1-3 day effect even if one existed.

The famous replication (Lachanski & Pav 2017) reached the same conclusion: the result
does not hold up. This is a **None / Mirage**.
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
# Study 256 -- Twitter-Mood
## For the quants: HAC lead-lag, the Granger-lag trap, permutation null, gross/net costs

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("## Positive control: the engine recovers a planted lead-lag, reads ~0 on the null"),
        code("""\
for knob in [0.0, 0.002, 0.004]:
    dp, _ = data.synthetic_series(n_days=200, predictive=knob, seed=256)
    r = st.leadlag_regression(dp, lag=1)
    print(f"  predictive={knob:.3f}: beta={r['beta']:+.5f}  HAC t={r['tstat']:+.2f}")
print()
print(f"Frozen: null t={R['pc_null_t']:+.2f}, mid t={R['pc_mid_t']:+.2f}, planted t={R['pc_hi_t']:+.2f}")
print("-> The machinery is truthful: it finds a lead-lag when one exists.")
"""),

        md("## Lead-lag regression sweep (the Granger-lag multiple-comparisons trap)"),
        code("""\
if HAVE_REAL:
    print("=== ret_{t+lag} ~ a + b * Calm_t  (HAC t) ===")
    for _, row in sweep.iterrows():
        print(f"  lag {int(row['lag'])}: beta={row['beta']:+.5f}  HAC t={row['tstat']:+.2f}  R2={row['r2']:.4f}  n={int(row['n'])}")
    best = sweep.iloc[sweep['tstat'].abs().idxmax()]
    print(f"  Best |t| across 5 lags: lag {int(best['lag'])}  t={best['tstat']:+.2f}")
else:
    for L in range(1, 6):
        print(f"  lag {L}: beta={R[f'beta{L}']:+.5f}  HAC t={R[f't{L}']:+.2f}")
print(f"\\nBonferroni bar for 5 lags at 5%: |t| ~ {R['bonf_bar']:.2f}. Nothing clears even the *uncorrected* bar.")
"""),

        md("## Permutation null: where does the real slope land under time-shuffled mood?"),
        code("""\
if HAVE_REAL:
    sn = st.shuffle_null(df_real, lag=1, n_shuffles=2000, seed=256)
    betas_mean, betas_std = sn['perm_beta_mean'], sn['perm_beta_std']
    real_beta, perm_p = sn['real_beta'], sn['perm_p']
    # reconstruct an approximate null band for the plot
    xs = np.linspace(betas_mean - 4*betas_std, betas_mean + 4*betas_std, 200)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(xs, np.exp(-0.5*((xs-betas_mean)/betas_std)**2), color=GREY, alpha=.3, label="shuffle null")
    ax.axvline(real_beta, color=RED, lw=2, label=f"real beta (p={perm_p:.2f})")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Lag-1 slope vs the time-shuffle null -- the real slope is unremarkable")
    ax.set_xlabel("slope beta"); ax.set_yticks([]); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()
    print(f"Real lag-1 beta={real_beta:+.5f}  t={sn['real_tstat']:+.2f}  permutation p={perm_p:.4f}")
else:
    print(f"Frozen permutation p-values: lag1={R['perm_p1']}, lag2={R['perm_p2']}, lag3={R['perm_p3']}")
    print("The real slope sits comfortably inside the null at every lag.")
"""),

        md("## Directional hit-rate vs the unconditional up-rate (the base-rate trap)"),
        code("""\
if HAVE_REAL:
    hr = st.directional_hitrate(df_real, lookback=20)
    print(f"Calm-rule hit-rate : {hr['hit_rate']*100:.1f}%")
    print(f"Unconditional up   : {hr['uncond_up_rate']*100:.1f}%")
    print(f"Edge over baseline : {(hr['hit_rate']-hr['uncond_up_rate'])*100:+.1f} pts   (n={hr['n']})")
else:
    print(f"Frozen: hit={R['hit_rate']}%  uncond={R['uncond_up']}%  edge=+{R['edge_pts']} pts")
print("\\nThe binary predictor inherits the market's upward drift -- it is not skill.")
"""),

        md("## Trading rule: gross/net, long-only vs long-short, with a 1-day execution lag"),
        code("""\
if HAVE_REAL:
    for short, label in [(False, "long/flat"), (True, "long/short")]:
        res = st.mood_strategy(df_real, lookback=20, allow_short=short, one_way_bps=5.0)
        sg, snet, sm = st.summarize(res['gross']), st.summarize(res['net']), st.summarize(res['mkt'])
        print(f"[{label}] gross {sg['mean']*252*100:+.1f}%/yr (t={sg['tstat']:+.2f}) | "
              f"net@5bps {snet['mean']*252*100:+.1f}%/yr (t={snet['tstat']:+.2f})")
    print(f"Buy & hold (window): {sm['mean']*252*100:+.1f}%/yr (t={sm['tstat']:+.2f})  <- the 2008 crash")
else:
    print(f"[long/flat ] gross {R['long_gross_ann']:+.1f}%/yr (t={R['long_gross_t']:+.2f}) | net {R['long_net_ann']:+.1f}%/yr (t={R['long_net_t']:+.2f})")
    print(f"[long/short] gross {R['ls_gross_ann']:+.1f}%/yr (t={R['ls_gross_t']:+.2f}) | net {R['ls_net_ann']:+.1f}%/yr (t={R['ls_net_t']:+.2f})")
    print(f"Buy & hold (window): {R['bh_ann']:+.1f}%/yr (t={R['bh_t']:+.2f})")
print("\\nThe long-short 'edge' is a net-short-into-the-crash artefact; it does NOT clear t>=2.")
print("Shorts pay borrow (charged); costs are one-way bps x |dpos| on NAV.")
"""),

        md("## The n=207 power problem"),
        code("""\
# With ~207 daily obs and a true daily R^2 of ~1%, what t-stat could we even hope for?
import numpy as np
n = R['n_days']
for true_r2 in [0.005, 0.01, 0.02]:
    # approx slope t = sqrt(R2/(1-R2) * (n-2))
    t_exp = np.sqrt(true_r2/(1-true_r2) * (n-2))
    print(f"  true daily R2={true_r2:.3f}: expected |t| ~ {t_exp:.2f}")
print(f"\\nObserved lag-1 R2={R['r2_1']:.4f} -> expected |t|<1. The window is simply")
print("too short to detect a faint 1-3 day lead-lag, even if one truly existed.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 256 -- Twitter-Mood ===")
print()
print(f"Signal: NONE")
print(f"  Lag-1 lead-lag HAC t = +{R['t1']:.2f}; no lag in the 1-5 sweep clears t>=2")
print(f"  Permutation p = {R['perm_p1']} (lag 1); directional edge = +{R['edge_pts']} pts over base rate")
print(f"  n = {R['n_days']} business days -- underpowered; mood tape is a curated proxy")
print()
print(f"Tradability: MIRAGE")
print(f"  Long-only rule flat after costs (t={R['long_net_t']:.2f})")
print(f"  Long-short +{R['ls_net_ann']:.0f}%/yr is a 2008-crash net-short artefact (t={R['ls_net_t']:.2f}, < 2)")
print()
print("Honesty: curated-proxy mood tape (not the GPOMS feed); price-only S&P;")
print("  one-day execution lag; one-way costs x NAV; shorts pay borrow -- all applied.")
print()
print("Bottom line: None/Mirage -- the canonical 'social-media alpha' claim is a")
print("  mirage that the famous replication (Lachanski & Pav 2017) already buried.")
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
