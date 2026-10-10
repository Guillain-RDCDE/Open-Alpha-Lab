"""Notebook builder for Study 1018 — Correlations Go to One.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    cd notebooks
    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both walk the seven desk beats (METHODOLOGY.md) at two altitudes. The headline numbers quoted in
prose live in ONE place — the ``RESULTS`` dict below, mirroring ``docs/results.md`` (rerun
``examples/verify.py`` and update it together). The executed cells recompute everything on the
same pinned real tape, with fewer simulations and bootstrap replicates so each notebook runs in
well under three minutes; their numbers agree with the dict to Monte-Carlo error.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# The real-tape headline — copied from docs/results.md (verify.py, as-of 2022-11-30)
# --------------------------------------------------------------------------- #
RESULTS = {
    "as_of": "2022-11-30", "fp_stocks": "bbc82157e299", "fp_index": "e223c581ae22",
    "signal": "Real", "trad": "Fragile",
    "naive_calm": 0.206, "naive_crisis": 0.466, "artefact": -0.001, "genuine": 0.262,
    "genuine_lo": 0.147, "genuine_hi": 0.345, "genuine_t": 5.0, "n_regimes_pos": 6,
    "contemp_naive": 0.320, "contemp_artefact": 0.078,
    "fr_crisis": 0.160, "var_ratio": 11.2, "idio_ratio": 1.56, "const_beta_crisis": 0.698,
    "max_rolling_corr": 0.735, "full_corr": 0.302,
    "asym": -0.019, "asym_lo": -0.074, "asym_hi": 0.067,
    "asym_devol": 0.139, "asym_devol_lo": 0.073, "asym_devol_hi": 0.204,
    "exc_down_1": 0.49, "exc_up_1": 0.51, "exc_gauss_1": 0.21,
    "vol_calm": 0.124, "vol_crisis": 0.371, "vol_pred": 0.260,
    "shortfall": 0.429, "shortfall_lo": 0.260, "shortfall_hi": 0.543,
    "dr_calm": 2.14, "dr_crisis": 1.44, "episode_ratio_median": 1.31, "n_episodes": 10,
}
R_ = RESULTS

BADGES = ("![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
          "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617"
          "?style=flat-square)")

BOOT = """\
import sys, os, warnings
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
warnings.filterwarnings("ignore")
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.dpi"] = 80
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from goestoone import data, strategy as st
R, m = data.load_returns()
print(f"as-of {data.AS_OF} | {R.shape[1]} stocks (survivor sample), {len(R)} days, "
      f"{R.index[0].date()} -> {R.index[-1].date()}")
print(f"fingerprints: stocks {data.fingerprint(data.load_prices())}, "
      f"index {data.fingerprint(data.load_index())}")
"""


def md(text):
    return new_markdown_cell(text.strip("\n").rstrip())


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(f"""
# Correlations Go to One \U0001F517
### "In a crisis, everything falls together." Partly true — and partly a trick of the ruler.

{BADGES}

**Beat 0 · Verdict (real tape).** When markets get stormy, these 20 big US stocks really do start
moving together: their average correlation climbs from **{R_['naive_calm']:.2f}** in calm markets to
**{R_['naive_crisis']:.2f}** in turbulent ones, and almost none of that is the statistical illusion
that usually inflates this number. But it never goes anywhere near one, and diversification
shrinks rather than fails. Numbers from [`docs/results.md`](../docs/results.md), as-of
{R_['as_of']}.

> \U0001F4D3 **This is the plain-language layer.** The artefact benchmark, the bootstrap and the
> Forbes–Rigobon algebra are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> ⚠️ **Not investment advice.** Every chart below is computed by the code beside it, on the
> pinned real tape.
"""),
        code(BOOT),
        md("""
## 1 · The claim

*"In a crisis all correlations go to one — diversification fails exactly when you need it."*
It's one of the most repeated lines in finance, and practitioners have real scars behind it: in
2008 and again in March 2020, assets that were supposed to zig while others zagged fell
together. Sébastien Page and Robert Panariello put the strong version in a 2018 *Financial
Analysts Journal* article titled, simply, **"When Diversification Fails"**.

The strongest version of the claim has three parts: (a) correlations rise sharply in crises,
(b) that rise is real, not a measurement quirk, and (c) it is big enough to wreck the benefit of
holding many stocks.
"""),
        md("""
## 2 · So what?

If it were fully true, "diversify" would be a fair-weather promise: the protection would vanish
exactly in the months that decide a portfolio's fate, and holding 20 stocks would be no safer
than holding one when it mattered. If it were *false*, risk models could keep using calm-period
correlations and only worry about volatility.

The truth decides how much a sensible investor should over-estimate risk "just in case".
"""),
        md("""
## 3 · How we'd know

There's a catch that most retellings miss. **If you pick the stormy days, you will measure a
higher correlation even if nothing about the relationship changed.** Big common moves are what
*make* a day stormy, and big common moves are what correlation counts. So we:

1. measure the naive number six different ways (four use only *yesterday's* information to say
   "this is a crisis day");
2. build a fake market where the correlation is **constant by construction** but volatility
   swings exactly like the real one — and run the identical measurement on it. Whatever it
   reports is pure illusion;
3. call the difference "genuine" and put an error bar on it;
4. ask what the genuine part does to someone who simply holds all 20 stocks.

**What would make it a mirage:** a genuine increase indistinguishable from zero, or a portfolio
whose crisis risk is fully explained by volatility alone.
"""),
        md("""
## 4 · The teardown

### 4.1 How correlated are these stocks, over time?
"""),
        code("""
roll = st.rolling_avg_corr(R, window=63, step=5)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(roll.index, roll, color="#1f6feb", lw=1.2)
ax.axhline(1.0, color="#c0392b", ls="--", lw=1.2, label="'correlations go to one'")
ax.axhline(roll.median(), color="k", ls=":", lw=1, label=f"median {roll.median():.2f}")
ax.set_ylim(0, 1.05); ax.set_ylabel("average pairwise correlation, 63-day window")
ax.set_title("20 large US stocks: correlation spikes in every crisis — and never reaches one")
ax.legend(loc="upper left")
plt.show()
print(f"highest 63-day reading: {roll.max():.2f} (window ending {roll.idxmax().date()})")
"""),
        md("""
Every spike lines up with a crisis — 1998, 2002, 2008, 2011, 2020. The spikes are real. The
red line at one is never threatened.

### 4.2 The naive table
"""),
        code("""
labels = st.all_labels(m)
rows = []
for k, lab in labels.items():
    rc = st.regime_corr(R, lab)
    rows.append({"how 'crisis' is defined": st.REGIMES[k]["label"],
                 "uses only the past?": "yes" if st.REGIMES[k]["past_only"] else "NO",
                 "calm": rc["calm"], "crisis": rc["crisis"], "gap": rc["diff"]})
naive = pd.DataFrame(rows)
naive
"""),
        md("""
### 4.3 The illusion, measured

Now the fake market: every stock keeps its own real volatility pattern, but the correlation
between their surprises is held **constant**. Run the same six measurements on it.
"""),
        code("""
fit = st.fit_ccc_garch(R, m)
art = st.artefact_benchmark(fit, n_sims=6, exceed=False)
tbl = pd.DataFrame({
    "real gap": [st.regime_corr(R, labels[k])["diff"] for k in labels],
    "illusion (constant-corr world)": [art["regimes"][k]["diff_mean"] for k in labels],
}, index=list(labels))
tbl["genuine"] = tbl["real gap"] - tbl["illusion (constant-corr world)"]
ax = tbl[["illusion (constant-corr world)", "genuine"]].plot.barh(
    stacked=True, color=["#8b949e", "#1f6feb"], figsize=(10, 4.5))
ax.set_xlabel("rise in average correlation, crisis minus calm")
ax.set_title("Grey = what a constant-correlation market would also show")
plt.show()
tbl
"""),
        md(f"""
The grey slivers are the illusion. When "crisis" is defined from *yesterday's* information it is
essentially zero; when it is defined from the month's own turbulence — the way the folklore
usually measures — it is about a quarter of the gap. Either way the blue part, the genuine rise,
is large: **{R_['genuine']:+.2f}** on the headline definition, with an error bar of
[{R_['genuine_lo']:+.2f}, {R_['genuine_hi']:+.2f}].

> \U0001F52C **For the quants.** 40 constant-correlation GARCH paths in the pinned run (6 here),
> circular block bootstrap with 63-day blocks, labels carried with the days. See notebook 02 §3.
"""),
        md("""
### 4.4 Does it hurt someone holding all 20?

Take an equal-weight book. Ask: in crisis periods, how risky *was* it, versus how risky it
*would have been* if each stock's volatility had jumped but correlations had stayed at their calm
level?
"""),
        code("""
ps = st.portfolio_stats(R, labels["past21"])
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(["calm\\nmarkets", "crisis: if only\\nvolatility rose", "crisis:\\nwhat happened"],
              [ps["vol_calm"] * 100, ps["vol_pred"] * 100, ps["vol_crisis"] * 100],
              color=["#2ea44f", "#dab617", "#c0392b"])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.0f}%",
            ha="center")
ax.set_ylabel("annualised volatility of the 20-stock book, %")
plt.show()
print(f"correlation shortfall: {ps['shortfall']:+.0%}   "
      f"diversification ratio {ps['dr_calm']:.2f} calm -> {ps['dr_crisis']:.2f} crisis "
      f"(1.00 = no diversification at all)")
"""),
        md("""
The jump from yellow to red is the price of correlation rising. It is real and large. But the
diversification ratio — how much risk the mix removes compared to the average stock — falls
from about 2.1 to about 1.4, not to 1.0. **Diversification shrinks; it does not fail.**
"""),
        code("""
ep = st.episode_table(R)
ep[["days", "corr_before", "corr_during", "ratio", "dr_before", "dr_during"]]
"""),
        md("""
Each named crisis, with the yardstick a holder could actually have had: correlations from the
year *before*. Every row's `ratio` is above 1 — the book was riskier than "old correlations, new
volatilities" predicted — and every `dr_during` stays above 1.
"""),
        md(f"""
## 5 · The verdict

**Signal: {R_['signal']}.** The crisis rise in correlation is genuine:
{R_['naive_calm']:.2f} → {R_['naive_crisis']:.2f}, essentially none of it the statistical illusion when
crises are defined from past data, positive under all six definitions. Correlations do **not**
go to one — the highest 63-day reading in 33 years was {R_['max_rolling_corr']:.2f}.

**Tradability: {R_['trad']}.** A diversified stock book was about {R_['shortfall']:.0%} riskier in
crises than volatility alone predicts, and its diversification ratio fell from {R_['dr_calm']:.2f}
to {R_['dr_crisis']:.2f}. That matters for risk budgets; it is not a trade, so it can't be
"investable" — and it is not the collapse the slogan promises.

> \U0001F52C **For the quants.** A twist: by the Forbes–Rigobon yardstick there is *no contagion*
> at all — the whole rise is a louder market factor. That doesn't make it fake; it makes it
> undiversifiable. Notebook 02 §4.
"""),
        md("""
## 6 · Could you trade it?

Not directly — there is no signal here that tells you what to buy. What you *can* do:

- **Stress the correlations, not just the volatilities.** Here the book's real crisis risk came
  out about 40% above what a model scaling calm covariances by crisis volatilities predicted.
- **Don't expect more stocks to fix it.** The rise is in the common factor; adding names from the
  same market adds the same factor.
- **Diversify across things that don't share that factor** (bonds, cash, explicit hedges) — and
  test *those* correlations in crises too. That's a different study.
"""),
        md("""
## 7 · Going further \U0001F6AA

- Run the same machinery on **cross-asset** pairs (stocks vs Treasuries) where the sign of the
  correlation itself flips — the real "diversification failure" story may live there.
- Replace the Gaussian constant-correlation benchmark with **fat-tailed** shocks and see how much
  the contemporaneous illusion grows.
- The survivor panel is a known limit: a point-in-time constituent list would let a contributor
  check the reasoning about its direction.
- Neighbours: [578](../../578-cross-asset-correlation-regime/) asks whether high correlation
  *predicts* returns (it doesn't, the way people think);
  [974](../../974-diversification-saturation/) prices the *k*-th asset in calm averages.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(f"""
# Correlations Go to One — a quantitative teardown \U0001F52C
### Conditional correlation · the BGL/Forbes–Rigobon artefact · a constant-correlation GARCH null · exceedances · the holder's arithmetic

{BADGES}

**Beat 0 · Verdict (real tape).** Past-only crisis regime: average pairwise correlation
{R_['naive_calm']:.3f} → {R_['naive_crisis']:.3f}; constant-correlation GARCH artefact
{R_['artefact']:+.3f}; genuine **{R_['genuine']:+.3f}**, circular block-bootstrap 95% CI
[{R_['genuine_lo']:+.3f}, {R_['genuine_hi']:+.3f}] (bootstrap *t* ≈ {R_['genuine_t']:.1f}). FR-adjusted
crisis correlation {R_['fr_crisis']:.3f} < calm: interdependence, not contagion. Equal-weight book:
crisis vol {R_['vol_crisis']:.1%} vs {R_['vol_pred']:.1%} predicted → correlation shortfall
**{R_['shortfall']:+.1%}** [{R_['shortfall_lo']:+.1%}, {R_['shortfall_hi']:+.1%}], DR {R_['dr_calm']:.2f} →
{R_['dr_crisis']:.2f}. Headline numbers: [`docs/results.md`](../docs/results.md) (as-of
{R_['as_of']}, fingerprints `{R_['fp_stocks']}` / `{R_['fp_index']}`; 40 sims, 400 bootstrap
replicates). Cells below re-run on the same tape with fewer replications.

> ⚠️ **Not investment advice.** Survivor sample of 20 large caps (skfolio `sp500_dataset`); the
> S&P 500 series is a **price** index used only as the state variable.
>
> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back into intuition.
> The plain-language story is in [01_for_the_curious.ipynb](01_for_the_curious.ipynb).
"""),
        code(BOOT),
        md("""
## 1 · The claim, steelmanned

"Correlations go to one in a crisis" (Page & Panariello 2018 for a practitioner statement;
Longin & Solnik 2001 and Ang & Chen 2002 for the academic evidence of tail and downside
dependence). Hypotheses, fixed before the run:

- **H₁ (naive).** Average pairwise correlation is higher in crisis than calm regimes.
- **H₂ (genuine).** The gap survives subtracting what the identical procedure reports under a
  constant-correlation model with the tape's own volatility dynamics. *Signal stamp hinges on
  this:* Real iff the headline (past-21-day vol) genuine gap has a block-bootstrap 95% CI above
  zero and is positive in ≥ half of six regime definitions.
- **H₃ (mechanism).** Forbes–Rigobon: is the rise explained by a louder common factor at constant
  betas?
- **H₄ (tails).** Exceedance correlations exceed the Gaussian benchmark, more on the downside.
- **H₅ (holder).** An equal-weight book's crisis vol exceeds the vol implied by calm correlations
  and crisis volatilities. *Tradability:* Fragile iff that shortfall is ≥ 5% with CI above 0;
  otherwise Mirage. Investable is unreachable — nothing here is a trading rule.
"""),
        md("""
## 2 · Why the naive number is biased — Boyer, Gibson & Loretan in one line

For ``y = βx + ε`` with the subsample selected on ``x`` alone and ``k = Var(x|A)/Var(x)``:
``ρ_A = ρ√k / √(1 + ρ²(k−1))``. The Forbes–Rigobon adjustment is its exact inverse.
"""),
        code("""
k = np.linspace(0.2, 20, 200)
fig, ax = plt.subplots(figsize=(9, 4.5))
for rho in (0.1, 0.2, 0.4, 0.6):
    ax.plot(k, [st.bgl_conditional_corr(rho, kk) for kk in k], lw=2, label=f"true rho {rho}")
ax.axvline(1, color="k", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("variance ratio of the selected days (log)")
ax.set_ylabel("measured correlation"); ax.legend()
ax.set_title("Select stormy days and a constant correlation reads higher")
plt.show()
# Monte-Carlo check of the closed form
rng = np.random.default_rng(1018)
x = rng.normal(size=400_000); y = 0.3 * x + np.sqrt(1 - 0.09) * rng.normal(size=400_000)
sel = np.abs(x) > 1.5
print(f"selected |x|>1.5: measured {np.corrcoef(x[sel], y[sel])[0, 1]:.3f}, closed form "
      f"{st.bgl_conditional_corr(0.3, x[sel].var() / x.var()):.3f}, truth 0.300")
"""),
        md("""
> \U0001F4A1 **In plain words.** The S&P's variance on high-vol days is ~11× its calm level. At
> that ratio a true correlation of 0.2 would *read* about 0.55 if you selected on the market's
> own moves. That is why the folklore number cannot be taken at face value.

## 3 · The teardown

### 3.1 Naive gaps under six definitions
"""),
        code("""
labels = st.all_labels(m)
naive = pd.DataFrame({k: st.regime_corr(R, lab) for k, lab in labels.items()}).T
naive["past_only"] = [st.REGIMES[k]["past_only"] for k in naive.index]
naive
"""),
        md("""
### 3.2 The exact simulation benchmark — constant-correlation GARCH

Each stock and the S&P: GARCH(1,1) on the real tape; shocks Gaussian with the **constant**
correlation of the standardised residuals (Bollerslev 1990). The identical regime procedure on
simulated paths gives the artefact; the circular block bootstrap (63-day blocks, labels carried
with days) gives the interval for ``naive − artefact``.
"""),
        code("""
fit = st.fit_ccc_garch(R, m)
print(f"median alpha+beta {np.median([p['alpha'] + p['beta'] for p in fit['params']]):.3f}; "
      f"constant shock correlation (stocks) {fit['avg_corr_stocks']:.3f}")
art = st.artefact_benchmark(fit, n_sims=10, seed=1018, exceed=True)
Z = st.devolatilise(R, m, fit)
bt = st.bootstrap(R, m, labels, n_boot=120, block=63, seed=1018, exceed=True, Z=Z)
rows = []
for k in labels:
    a = art["regimes"][k]["diff_mean"]
    lo, hi = st.ci(bt[k]["diff"], centre_shift=a)
    rows.append({"regime": k, "naive": naive.loc[k, "diff"], "artefact": a,
                 "genuine": naive.loc[k, "diff"] - a, "ci_lo": lo, "ci_hi": hi,
                 "p": st.boot_p_positive(bt[k]["diff"], centre_shift=a)})
dec = pd.DataFrame(rows).set_index("regime")
fig, ax = plt.subplots(figsize=(10, 4.5))
y = np.arange(len(dec))
ax.errorbar(dec["genuine"], y, xerr=[dec["genuine"] - dec["ci_lo"], dec["ci_hi"] - dec["genuine"]],
            fmt="o", color="#1f6feb", capsize=4, label="genuine (95% block-bootstrap CI)")
ax.scatter(dec["naive"], y, marker="x", color="#c0392b", label="naive gap")
ax.scatter(dec["artefact"], y, marker="s", color="#8b949e", label="artefact (const-corr GARCH)")
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(dec.index)
ax.set_xlabel("crisis minus calm average pairwise correlation"); ax.legend(fontsize=9)
plt.show()
dec
"""),
        md("""
> \U0001F4A1 **In plain words.** Under a model whose conditional correlation never moves, a
> *past-only* regime reports ≈ 0 — the artefact needs the selection to see the day's own
> realisations. The contemporaneous month-volatility definition does see them, and manufactures
> roughly a quarter of its own gap. Every definition still leaves a large, significant genuine
> rise.

### 3.3 Forbes–Rigobon and the constant-beta benchmark
"""),
        code("""
rows = []
for k, lab in labels.items():
    fr = st.fr_panel(R, m, lab); cb = st.constant_beta_benchmark(R, m, lab)
    iv = st.idio_vol_ratio(R, m, lab)
    rows.append({"regime": k, "calm": fr["calm_raw"], "crisis": fr["crisis_raw"],
                 "crisis_FR_adj": fr["crisis_fr"], "const_beta_implies": cb["implied_crisis"],
                 "mkt_var_x": fr["var_ratio"], "idio_vol_x": iv["idio_vol_ratio"],
                 "beta_x": iv["beta_ratio"]})
frt = pd.DataFrame(rows).set_index("regime")
frt
"""),
        md("""
> \U0001F4A1 **In plain words.** Betas didn't change (×1.00). The market got ~3.3× more volatile;
> stock-specific risk only ~1.6×. Correlation is the common part's share of the total, so it rose.
> Forbes and Rigobon call that "interdependence, not contagion" and their adjustment pulls the
> crisis correlation *below* the calm one. Both statements are true at once: the transmission is
> unchanged, *and* the holder's diversification got worse. The constant-beta benchmark
> over-predicts (≈0.70) precisely because idiosyncratic risk rose too.

### 3.4 Exceedance correlations
"""),
        code("""
cur = st.exceedance_panel(R, m)
gau = st.gaussian_panel_curve(R, m)
curz = st.exceedance_panel(*Z)
fig, ax = plt.subplots(figsize=(10, 4.8))
for s, lab, c, ls in ((cur, "empirical (raw)", "#c0392b", "-"),
                      (curz, "empirical (GARCH-devolatilised)", "#1f6feb", "-"),
                      (gau, "Gaussian, same correlation", "k", "--"),
                      (art["curve"], "constant-corr GARCH", "#8b949e", ":")):
    for side in (s[s.index < 0], s[s.index > 0]):
        ax.plot(side.index, side.values, ls, color=c, lw=2, marker="o", ms=4,
                label=lab if side.index[0] < 0 else None)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("threshold theta (standard deviations); left = joint downside, right = joint upside")
ax.set_ylabel("exceedance correlation, stock vs S&P (avg of 20)"); ax.legend(fontsize=9)
plt.show()
print(f"asymmetry (down - up): raw {st.asymmetry(cur):+.3f} CI {st.ci(bt['_asym'])}, "
      f"devolatilised {st.asymmetry(curz):+.3f} CI {st.ci(bt['_asym_devol'])}, "
      f"const-corr GARCH {art['asym_mean']:+.3f}")
print(f"Ang-Chen H vs Gaussian: down {st.ang_chen_h(cur, gau, 'down'):.3f}, "
      f"up {st.ang_chen_h(cur, gau, 'up'):.3f}")
"""),
        md("""
> \U0001F4A1 **In plain words.** A normal distribution says correlation should *fade* in the
> tails (dashed line). Real stock-index tails stay at ~0.5 on both sides — far more joint
> extremes than a normal model allows. On raw daily returns the two sides are about equal,
> because the biggest rebounds happen inside the same volatile clusters as the biggest drops.
> Measured relative to the volatility of the moment, the downside shocks are clearly the more
> correlated ones — the Longin–Solnik / Ang–Chen asymmetry survives, in that form.

### 3.5 The holder
"""),
        code("""
rows = []
for k, lab in labels.items():
    ps = st.portfolio_stats(R, lab)
    a = art["regimes"][k]["shortfall_mean"]
    lo, hi = st.ci(bt[k]["shortfall"], centre_shift=a)
    rows.append({"regime": k, "vol_calm": ps["vol_calm"], "vol_crisis": ps["vol_crisis"],
                 "vol_pred": ps["vol_pred"], "shortfall": ps["shortfall"], "artefact": a,
                 "genuine": ps["shortfall"] - a, "ci_lo": lo, "ci_hi": hi,
                 "dr_calm": ps["dr_calm"], "dr_crisis": ps["dr_crisis"]})
port = pd.DataFrame(rows).set_index("regime")
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.barh(port.index, port["dr_calm"], color="#2ea44f", alpha=0.6, label="DR calm")
ax.barh(port.index, port["dr_crisis"], color="#c0392b", alpha=0.8, label="DR crisis")
ax.axvline(1, color="k", lw=1.5, ls="--", label="no diversification")
ax.axvline(np.sqrt(20), color="#8b949e", lw=1, ls=":", label="ceiling, 20 uncorrelated")
ax.set_xlabel("diversification ratio  sum(w*sigma) / sigma_p"); ax.legend(fontsize=9)
plt.show()
port
"""),
        code("""
ep = st.episode_table(R)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.barh(ep.index, ep["ratio"], color="#dab617")
ax.axvline(1, color="k", lw=1.2)
ax.set_xlabel("realised book vol / vol predicted from PRE-episode correlations + episode vols")
plt.show()
ep.round(3)
"""),
        md("""
> \U0001F4A1 **In plain words.** With the yardstick a holder actually had — last year's
> correlations — every episode was riskier than "same correlations, higher volatility" (ratios
> 1.14–1.55). The diversification ratio never touched 1.

### 3.6 Synthetic control — the machinery, not the market
"""),
        code("""
rows = []
for s in (0.0, 0.5, 1.0):
    Rs, ms, tr = data.synthetic_panel(n_assets=10, n_years=16, signal_strength=s, seed=1018)
    d = st.crisis_correlation_test(Rs, ms, n_sims=6, n_boot=100)
    mv = st.regime_corr(Rs, st.regime_labels(ms, "month_vol"))["diff"]
    rows.append({"strength": s, "planted": tr["planted_jump"], "naive": d["diff"],
                 "artefact": d["artefact"], "genuine": d["genuine"], "ci_lo": d["ci_lo"],
                 "ci_hi": d["ci_hi"], "fires": d["significant"], "month_vol_naive": mv})
pd.DataFrame(rows).set_index("strength")
"""),
        md(f"""
> \U0001F4A1 **In plain words.** Constant correlation (strength 0): the detector is quiet, but the
> contemporaneous month-vol measurement still shows a positive gap — the artefact, alone. Planted
> jump: the detector fires. A synthetic result proves the harness, never the market.

## 4 · The verdict

**Signal: {R_['signal']}.** Genuine past-only rise {R_['genuine']:+.3f}, CI
[{R_['genuine_lo']:+.3f}, {R_['genuine_hi']:+.3f}], positive in {R_['n_regimes_pos']}/6 definitions; the
artefact is ≈ 0 for past-only regimes and {R_['contemp_artefact']:+.3f} of {R_['contemp_naive']:+.3f}
for the contemporaneous month-vol one. FR-adjusted crisis correlation {R_['fr_crisis']:.3f}:
interdependence, not contagion. Max 63-day average correlation {R_['max_rolling_corr']:.2f} — never one.
Raw exceedance asymmetry {R_['asym']:+.3f} [{R_['asym_lo']:+.3f}, {R_['asym_hi']:+.3f}]; devolatilised
{R_['asym_devol']:+.3f} [{R_['asym_devol_lo']:+.3f}, {R_['asym_devol_hi']:+.3f}].

**Tradability: {R_['trad']}.** Correlation shortfall {R_['shortfall']:+.1%}
[{R_['shortfall_lo']:+.1%}, {R_['shortfall_hi']:+.1%}]; DR {R_['dr_calm']:.2f} → {R_['dr_crisis']:.2f};
episodes median {R_['episode_ratio_median']:.2f}×. Real for risk budgets, not a return stream.

## 5 · Could you trade it?

There is no position here. The usable output is a **risk-model correction**: scale crisis
covariances by both a volatility multiplier *and* a correlation stress (here, roughly +0.25 on
average pairwise correlation, or ×1.4 on equal-weight book vol beyond the volatility effect). A
past-only regime identifies the state with essentially no artefact, so the correction can be
applied in real time — but volatility forecasting already captures most of the timing, and the
residual is a capital-allocation input, not alpha.

## 6 · Going further

- **DCC** (Engle 2002) instead of regimes: a continuous conditional-correlation path, and a test of
  whether its crisis level beats the CCC null on the same tape.
- **Fat-tailed null**: Student-*t* shocks in the CCC benchmark widen the contemporaneous artefact;
  the past-only conclusion should not move.
- **Cross-asset** (stock–bond) correlations, where the sign itself changes by regime.
- **Survivorship**: a point-in-time constituent panel would test the reasoned direction of the bias.
- Neighbours on the desk: 578 (correlation regime as a *return predictor*), 1010 (estimation noise
  in correlation matrices), 502 (correlation as a cross-sectional sort), 974 (diversification
  saturation).
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


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
