"""Notebook builder for Study 1024 — The Drawdown You Were Promised.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    python notebooks/build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks walk the same seven desk beats (METHODOLOGY.md) at two altitudes: 01 is the
plain-language layer, 02 carries the inference, the decomposition and the robustness. The
verdict badges in the heroes are the ones the real run earned, and every real-tape number
quoted in prose lives in the single ``REAL`` dict below, copied from ``docs/results.md`` —
never typed by hand into a cell. Code cells recompute what they plot from the bundled tapes.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Real-tape numbers, mirrored from docs/results.md (examples/verify.py, as-of 2022-11-30).
REAL = {
    "as_of": "2022-11-30",
    "signal": "Real", "trad": "Mirage",
    "sp_rate": "22%", "sp_k": "6/27", "sp_p": "0.002",
    "nq_rate": "13%", "nq_k": "2/15", "nq_p": "0.171",
    "ff_rate": "15%", "ff_k": "12/81", "ff_p": "0.001",
    "pooled_rate": "16%", "pooled_n": "123", "overrun": "3.3×",
    "median_ratio": "0.68",
    "oracle_rate_sp": "15%",
    "mu0_rate": "8%", "boot_rate": "15%", "garch_rate": "14%",
    "k95_pooled": "1.63", "k95_min": "1.03", "k95_max": "1.92", "k95_range": "1.87",
    "k95_mu0": "1.19",
    "stocks_rate": "12%", "stocks_k95": "1.33", "stocks_k95_ci": "1.03–1.60",
    "stocks_worst": "GE", "stocks_worst_rate": "30%",
    "null_oracle": "5.8%", "null_plugin": "7.7%", "planted_plugin": "15.7%",
    "disc_err": "1.1%", "disc_monthly": "20%",
    "ff_expanding": "7% (p = 0.28)", "sp_lookbacks": "14–22%",
}

R = REAL

SIGNAL_WHY = (
    f"Written down from the preceding years only, the Gaussian 95% drawdown band was breached in "
    f"1-year non-overlapping windows on the S&P 500 **{R['sp_rate']}** ({R['sp_k']}, "
    f"p = {R['sp_p']}), the Nasdaq **{R['nq_rate']}** ({R['nq_k']}, p = {R['nq_p']}) and the "
    f"Fama-French market **{R['ff_rate']}** ({R['ff_k']}, p = {R['ff_p']}) — against a promised "
    f"5%, and **{R['pooled_rate']}** pooled. The machinery is calibrated where it should be "
    f"({R['null_oracle']} on an i.i.d. Gaussian tape), so the failure belongs to the tape: the "
    f"median year delivered only **{R['median_ratio']}×** the promised mean drawdown, yet the bad "
    f"years overran the 95th percentile {R['overrun']} as often as promised. Even with the whole "
    f"tape's mu and sigma in hindsight the S&P 500 band broke in {R['oracle_rate_sp']} of years. "
    f"Twenty survivor stocks — a floor — broke it in {R['stocks_rate']} of stock-years.")
TRAD_WHY = (
    f"Scaling the Gaussian takes **k95 = {R['k95_pooled']}×** pooled, but the multiplier a risk "
    f"manager would have needed wanders from {R['k95_min']} to {R['k95_max']} across tapes and "
    f"eras (max/min {R['k95_range']}, outside the pre-registered 1.5). Neither the block "
    f"bootstrap ({R['boot_rate']}) nor GARCH-t ({R['garch_rate']}) restored coverage on all three "
    f"tapes. The cheapest partial repair was a humbler mean — drift set to zero, "
    f"{R['mu0_rate']} pooled, k95 {R['k95_mu0']} — at the price of over-promising at 5–10 years. "
    f"No return stream, so Investable is not on offer.")
ONE = (f"Fed the trailing mean and volatility, the Gaussian drawdown promise was broken in "
       f"{R['pooled_rate']} of years instead of 5% — knowing mu and sigma is not knowing drawdown "
       f"risk — and no fixed multiplier (it wandered from 1.0× to 1.9×) and no off-the-shelf "
       f"model repairs it.")

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)")

BOOT = r"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
warnings.filterwarnings("ignore")
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from promised import data, strategy as st
print("real tapes available:", data.have_real(), "| as-of", data.AS_OF)
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
# The Drawdown You Were Promised \U0001F91E
### A risk model that knows the mean and the volatility claims to know how far you can fall. We checked the promise against 90 years of tape.

{BADGES}

Ask a risk system "how bad can a year get?" and, if it believes returns are a random walk with a
known average and a known volatility, it can answer exactly — there is a formula. This notebook
writes that answer down **before** each year, using only what was known at the time, and then
looks at what the year actually did.

> \U0001F4D3 **This is the plain-language layer.** Binomial tests, the overlapping-window
> bootstrap, the era splits and the synthetic calibration are in
> **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> ⚠️ **Not investment advice.** Every chart below is generated by the code beside it.
"""),
        code(BOOT),
        md(f"""
## The answer first \U0001F3AF

| Question | Answer |
|---|---|
| Is the "how bad can it get" number right? | No. The 95% worst case was beaten in **{R['pooled_rate']}** of years, not 5%. |
| Is it too gloomy or too cheerful? | Both: ordinary years are *calmer* than promised, bad years *much worse*. |
| Is the maths broken? | No — it is calibrated on a computer-made random walk. The market is not one. |
| Can you just multiply it by a safety factor? | The factor you'd need drifted from {R['k95_min']}× to {R['k95_max']}× depending on market and era. |
| Does a fancier model fix it? | Not reliably. Assuming *zero* average return helped more than modelling fat tails. |

**Signal: {R['signal']}** · **Tradability: {R['trad']}** — {ONE}
"""),
        md("""
## 1 · The claim

If prices wander like a random walk with a steady drift and a steady volatility, the largest
peak-to-trough fall over a year has a known distribution — worked out exactly by Magdon-Ismail,
Atiya, Pratap and Abu-Mostafa in 2004. Risk teams use it to set drawdown budgets ("we can
tolerate a 25% fall"), and the Calmar ratio is often normalised by it. The promise, at full
strength: **know the mean and the volatility, and you know the drawdown risk.**

Here is what that promise looks like for the S&P 500, using its own long-run average and
volatility:
"""),
        code(r"""
sp = data.load_sp500()
mu, sd = sp.mean(), sp.std()
fig, ax = plt.subplots(figsize=(10, 5))
for H, c in ((1, "#1f6feb"), (3, "#8250df"), (10, "#c0392b")):
    m = st.gbm_mdd(mu, sd, 252 * H, 4000)
    ax.hist(m * 100, bins=70, alpha=0.55, color=c, density=True,
            label=f"{H}-year horizon: median {np.median(m):.0%}, 95th pct {np.quantile(m, .95):.0%}")
ax.set_xlabel("largest peak-to-trough fall in the period, %"); ax.set_ylabel("density")
ax.set_title(f"The promise: S&P 500 as a random walk (drift {mu*252:.1%}/yr, vol {sd*np.sqrt(252):.1%})")
ax.legend(); plt.show()
"""),
        md("""
## 2 · So what?

A drawdown budget is the number a pension board, a fund's investors or a trader's risk manager
signs off on. If the "95% worst case" is breached once in twenty years, everyone planned for it.
If it is breached one year in six, the plan is fiction — and the breach comes precisely in the
year capital is most expensive to raise.

## 3 · How we'd know

Before every calendar year, take only the **previous five years** of daily data, estimate the
average and the volatility, and write down the promised range of falls for the year ahead. Then
check: was the year's actual fall worse than the promised **95% worst case**? If the model is
right, that should happen in about **1 year in 20**. We do it on three market tapes — the S&P
500 (1995–2021), the Nasdaq (2004–2018) and the whole US stock market from Fama and French
(1937–2017, monthly) — and on twenty big individual stocks.

**What would make us say "the promise holds":** a breach rate statistically indistinguishable
from 5%. One detail first: how often you *look* at prices changes the answer.
"""),
        code(r"""
d = st.discretisation_table(0.0, 0.16, 1.0, steps_per_year=(12, 52, 252, 252 * 16), n_sims=3000)
fig, ax = plt.subplots(figsize=(9, 4.5))
lbl = ["monthly", "weekly", "daily", "16× a day"]
ax.bar(lbl, d["mean"] * 100, color=["#8b949e", "#8b949e", "#1f6feb", "#8250df"])
ax.set_ylabel("average worst fall in a year, %")
ax.set_title("The same random walk, checked at different frequencies")
plt.show()
print(f"checking monthly hides {1 - d['mean'].iloc[0] / d['mean'].iloc[2]:.0%} of the average daily-close drawdown")
"""),
        md(f"""
The troughs between your observations don't show up in the drawdown. So the promise must be
computed at the same frequency the account is marked — daily for the indices, monthly for the
Fama-French series. Every number here does that.

## 4 · The teardown

### The S&P 500, one year at a time
"""),
        code(r"""
cov = st.coverage(sp, "sp500", 252, (1,), 5, ("gauss",), n_sims=4000)
fig, ax = plt.subplots(figsize=(12, 5.5))
x = cov["start"].to_numpy()
ax.fill_between(x, 0, cov["p_q95"] * 100, color="#1f6feb", alpha=0.15, step="mid",
                label="promised 95% worst case (written down in advance)")
ax.plot(x, cov["p_mean"] * 100, "--", color="#1f6feb", label="promised average fall")
br = cov["breach95"].to_numpy()
ax.bar(x[~br], cov["realised"][~br] * 100, width=0.6, color="#8b949e", label="actual fall")
ax.bar(x[br], cov["realised"][br] * 100, width=0.6, color="#c0392b", label="actual fall — promise broken")
ax.set_ylabel("largest peak-to-trough fall in the year, %"); ax.legend(loc="upper left")
ax.set_title(f"S&P 500: the promise was broken in {br.sum()} of {len(br)} years (5% would be ~1)")
plt.show()
"""),
        md(f"""
Look at *when* the red bars come: 1998, 2001, 2002, 2008, 2018, 2020. Most follow a stretch of
calm, rising markets, so the trailing five years said "low volatility, high average return" — and the
promise was at its most optimistic exactly when it was about to be tested. In between, the grey
bars sit well **below** the dashed line: in the median year the market fell only
**{R['median_ratio']}×** the promised average. The promise is too gloomy most of the time and far
too cheerful when it matters.

### Across all three tapes
"""),
        code(r"""
tapes = {t: data.load_tape(t) for t in data.INDEX_TAPES}
LB = {"sp500": 5, "nasdaq": 5, "ff_market": 10}
rows = []
for t, r in tapes.items():
    c = st.coverage(r, t, data.PERIODS_PER_YEAR[t], (1,), LB[t],
                    ("gauss", "gauss_mu0", "gauss_oracle", "boot", "garch_t"), n_sims=2000)
    rows.append(c)
one = pd.concat(rows, ignore_index=True)
rate = one.pivot_table(index="tape", columns="model", values="breach95", aggfunc="mean")
names = {"gauss": "random walk\n(trailing avg & vol)", "gauss_mu0": "random walk,\naverage set to 0",
         "gauss_oracle": "random walk,\nhindsight avg & vol", "boot": "reshuffled\nhistory",
         "garch_t": "volatility model\n(GARCH-t)"}
fig, ax = plt.subplots(figsize=(12, 5))
w = 0.27
for i, t in enumerate(data.INDEX_TAPES):
    ax.bar(np.arange(5) + (i - 1) * w, rate.loc[t, list(names)] * 100, width=w,
           label={"sp500": "S&P 500", "nasdaq": "Nasdaq", "ff_market": "US market (Fama-French)"}[t])
ax.axhline(5, color="k", ls="--", lw=1.5, label="what a correct promise gives (5%)")
ax.set_xticks(range(5)); ax.set_xticklabels(list(names.values()))
ax.set_ylabel("% of years the 95% worst case was broken"); ax.legend(fontsize=9)
plt.show()
"""),
        md(f"""
The plain random walk breaks its promise far too often everywhere. Two things that sound like
they should fix it don't, reliably: reshuffling the last five years of real returns (it just
replays the calm), and a volatility model that knows today's turbulence (it doesn't know next
year's). What helped most was the boring move — **assume the average return is zero**: the
trailing average is what makes the promise most cheerful right before a fall.

> \U0001F52C **For the quants.** Exact binomial p-values on non-overlapping windows: S&P 500
> {R['sp_p']}, Nasdaq {R['nq_p']}, Fama-French {R['ff_p']}. Overlapping 3/5/10-year windows
> with a moving-block bootstrap and the era-by-era multipliers are in notebook 02.

## 5 · The verdict

**Signal: {R['signal']}.** {SIGNAL_WHY}

**Tradability: {R['trad']}.** {TRAD_WHY}

## 6 · Could you use it?

The tempting fix is a safety factor: "take the textbook number and multiply by 1.5".
"""),
        code(r"""
g = one[one["model"] == "gauss"].copy()
fig, ax = plt.subplots(figsize=(10, 4.5))
for t, c in zip(data.INDEX_TAPES, ("#1f6feb", "#8250df", "#2ea44f")):
    gt = g[g["tape"] == t]
    ratio = np.sort((gt["realised"] / gt["p_q95"]).to_numpy())
    ax.plot(ratio, np.arange(1, len(ratio) + 1) / len(ratio) * 100, lw=2.5, color=c,
            label=f"{t}: factor needed {st.multiplier(gt['realised'], gt['p_q95']):.2f}×")
ax.axhline(95, color="k", ls="--", lw=1)
ax.set_xlabel("actual fall ÷ promised 95% worst case"); ax.set_ylabel("% of years at or below")
ax.legend(); plt.show()
"""),
        md(f"""
Pooled, you would have needed about **{R['k95_pooled']}×**. But split by market and era and the
factor ran from **{R['k95_min']}×** to **{R['k95_max']}×** — set it in calm decades and the next
crisis blows through it. For single stocks (and these 20 are *survivors*, the ones that made it
to 2022) the promise broke in {R['stocks_rate']} of stock-years; {R['stocks_worst']} broke it in
{R['stocks_worst_rate']} of its years. A drawdown budget built on mean and volatility alone is a
number to be stress-tested, not trusted.

## 7 · Going further \U0001F6AA

- **Drawdown is a path property, volatility is not.** Two years with the same volatility can
  have very different drawdowns, depending on whether the bad days arrive together. Any model
  that ignores that clustering under-promises.
- **Watch the average.** A trailing average return is most optimistic just before falls. A
  zero-drift promise is cruder and was better calibrated at one year here.
- **Fork it.** Try a regime-switching model, or implied volatility instead of trailing
  volatility, and see if *any* model hits 5% on all three tapes.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(f"""
# The Drawdown You Were Promised — a quantitative teardown \U0001F52C
### Exact discretely-sampled GBM · ex-ante coverage of the MDD band · bootstrap and GARCH-t challengers · the multiplier and its stability

{BADGES}

Companion to the [notebook for the curious](01_for_the_curious.ipynb). §1 shows the simulation
reproduces Magdon-Ismail et al. (2004) and why sampling frequency matters; §3 is the
pre-registered test; §4 decomposes the failure; §5 prices the fix; §7 checks the machinery on a
synthetic tape where the answer is known.

> ⚠️ **Not investment advice.** Real numbers are from `quantlab.bundled` tapes (SHA-256 pinned),
> as-of {R['as_of']}; the fingerprinted run is [`docs/results.md`](../docs/results.md).
>
> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT),
        md(f"""
## Verdict, up front

| Axis | Stamp | Why |
|---|---|---|
| **Signal** | {R['signal']} | {SIGNAL_WHY} |
| **Tradability** | {R['trad']} | {TRAD_WHY} |

> **In one sentence:** {ONE}

## 0 · Hypotheses and the pre-registered rule

- **H₁ (coverage).** With mu, sigma estimated on the preceding 5 years (10 for monthly), the
  1-year 95% MDD band of a discretely-sampled GBM is breached more than 5% of the time —
  exact one-sided binomial test on non-overlapping calendar windows, per tape. **Real** needs
  p < 0.05 on two of the three index tapes *and* a calibrated synthetic null.
- **H₂ (decomposition).** The failure survives handing the model the whole-tape moments
  (shape/state, not estimation).
- **H₃ (challengers).** A stationary block bootstrap or a GARCH(1,1)-t restores coverage on all
  three tapes (p ≥ 0.05 and rate ≤ 10%).
- **H₄ (multiplier).** The required factor k95 on the Gaussian 95th percentile is stable
  (max/min ≤ 1.5 across tapes and halves). **Fragile** if H₃ or H₄ holds; else **Mirage**.

## 1 · The promise and the discretisation
"""),
        code(r"""
d = st.discretisation_table(0.0, 0.16, 1.0, steps_per_year=(12, 52, 252, 252 * 4, 252 * 16), n_sims=3000)
print(d.round(4).to_string())
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.semilogx(d.index, d["mean_log"], "o-", lw=2.5, label="simulated E[MDD], log units")
ax.axhline(d["continuous_formula_log"].iloc[0], color="#c0392b", ls="--",
           label="Magdon-Ismail et al. (2004): √(π/2)·σ·√T")
ax.set_xlabel("observations per year"); ax.set_ylabel("expected max drawdown (log)")
ax.legend(); plt.show()
"""),
        md(f"""
> \U0001F4A1 **In plain words.** The exact simulation converges on the textbook formula
> (gap {R['disc_err']} at 16 steps a day). But a monthly-marked account sees ~{R['disc_monthly']}
> less drawdown than a daily-marked one: the formula must be matched to the marking frequency,
> so every promise below is simulated at the tape's own frequency.

## 2 · The tapes
"""),
        code(r"""
tapes = {t: data.load_tape(t) for t in data.INDEX_TAPES}
LB = {"sp500": 5, "nasdaq": 5, "ff_market": 10}
rows = []
for t, r in tapes.items():
    ppy = data.PERIODS_PER_YEAR[t]
    rows.append({"tape": t, "label": data.LABELS[t], "first": r.index[0].date(),
                 "last": r.index[-1].date(), "n": len(r), "mu_ann (log)": r.mean() * ppy,
                 "vol_ann": r.std() * np.sqrt(ppy), "excess kurtosis": r.kurt(),
                 "sha": data.sha_pin(t), "fingerprint": data.fingerprint(r)})
pd.DataFrame(rows).set_index("tape")
"""),
        md("""
## 3 · The ex-ante test — non-overlapping calendar windows

For each window the five models are estimated on the preceding years only; 4,000 paths for the
Gaussians at 1–3 years. (5- and 10-year windows are in `docs/results.md`; with 1–16 windows
each they are anecdotes.)
"""),
        code(r"""
cov = pd.concat([st.coverage(r, t, data.PERIODS_PER_YEAR[t], (1, 3), LB[t], st.MODELS, n_sims=4000)
                 for t, r in tapes.items()], ignore_index=True)
summ = st.summarise(cov)
summ[summ["model"] == "gauss"][["tape", "horizon", "n", "breaches95", "rate95", "p_greater",
                                "rate99", "median_ratio_mean", "mean_pit", "k95"]]
"""),
        code(r"""
g = cov[(cov["model"] == "gauss") & (cov["horizon"] == 1)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, t in zip(axes, data.INDEX_TAPES):
    p = g[g["tape"] == t]["pit"]
    ax.hist(p, bins=10, range=(0, 1), color="#1f6feb", alpha=0.75, density=True)
    ax.axhline(1, color="k", ls="--")
    ax.set_title(f"{t}: PIT of realised MDD (KS p = {st.ks_uniform_pit(p):.3f})", fontsize=10)
    ax.set_xlabel("share of promised paths with a smaller drawdown")
plt.show()
"""),
        md(f"""
> \U0001F4A1 **In plain words.** If the promise were right, these histograms would be flat. They
> pile up on the **left** (most years calmer than promised) *and* carry excess mass at the far
> right (the breaches). Pooled 1-year breach rate {R['pooled_rate']} of {R['pooled_n']} windows,
> {R['overrun']} the nominal rate; median realised/promised mean {R['median_ratio']}. That
> combination is a shape failure — the right scale would not fix it.

### 3.1 Overlapping windows, block-bootstrap inference
"""),
        code(r"""
ov = []
for t, r in tapes.items():
    c = st.coverage(r, t, data.PERIODS_PER_YEAR[t], (3, 5), LB[t], ("gauss",), step_years=1, n_sims=2000)
    for H, gh in c.groupby("horizon"):
        b = st.block_bootstrap_rate(gh.sort_values("start")["breach95"].to_numpy(), block=H)
        ov.append({"tape": t, "horizon": H, **b})
pd.DataFrame(ov)
"""),
        md("""
> \U0001F4A1 **In plain words.** Stepping windows one year at a time gives more of them, but they
> share most of their path; the moving-block bootstrap (blocks of H windows) keeps the shared
> crises together so the interval is honest.

## 4 · What is missing? Mean, shape, or volatility state
"""),
        code(r"""
one = cov[cov["horizon"] == 1]
tab = st.summarise(one)[["tape", "model", "n", "rate95", "p_greater", "p_less", "median_ratio_mean", "k95"]]
fig, ax = plt.subplots(figsize=(12, 4.8))
piv = tab.pivot(index="model", columns="tape", values="rate95").loc[list(st.MODELS)]
piv.plot.bar(ax=ax, rot=0, color=["#1f6feb", "#8250df", "#2ea44f"])
ax.axhline(0.05, color="k", ls="--", lw=1.5)
ax.set_ylabel("breach rate of the 95% band (1-year)")
ax.set_xticklabels([st.MODEL_LABELS[m].replace(", ", ",\n") for m in piv.index], fontsize=8)
plt.show()
tab
"""),
        md(f"""
> \U0001F4A1 **In plain words.** Three separate failures, sized:
> * **The trailing mean.** Setting mu = 0 drops the pooled breach rate to {R['mu0_rate']}. A
>   5-year trailing mean is highest right after long rallies, i.e. right before the falls.
> * **Shape and regime.** Even with whole-tape (hindsight) moments the S&P 500 band broke in
>   {R['oracle_rate_sp']} of years.
> * **Neither challenger is a cure.** The bootstrap ({R['boot_rate']} pooled) replays the
>   lookback's calm; GARCH-t ({R['garch_rate']}) starts from today's variance but mean-reverts it
>   within months, so it cannot anticipate a regime that has not started.

## 5 · The multiplier, and whether it is stable
"""),
        code(r"""
g1 = one[one["model"] == "gauss"].copy()
rows = []
for t, gt in g1.groupby("tape", sort=False):
    yrs = np.sort(gt["start"].unique()); cut = yrs[len(yrs) // 2]
    for lab, ge in (("all", gt), (f"<{cut}", gt[gt["start"] < cut]), (f">={cut}", gt[gt["start"] >= cut])):
        lo, hi = st.multiplier_ci(ge["realised"], ge["p_q95"], n_boot=500)
        rows.append({"tape": t, "era": lab, "n": len(ge),
                     "k95": st.multiplier(ge["realised"], ge["p_q95"]), "lo": lo, "hi": hi})
mt = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(11, 4.5))
y = np.arange(len(mt))
ax.errorbar(mt["k95"], y, xerr=[mt["k95"] - mt["lo"], mt["hi"] - mt["k95"]], fmt="o", color="#c0392b")
ax.axvline(1, color="k", lw=1); ax.set_yticks(y); ax.set_yticklabels(mt["tape"] + " " + mt["era"])
ax.set_xlabel("k95: factor on the Gaussian 95th percentile for 5% breaches (90% bootstrap interval)")
plt.show()
print(f"max/min across tapes and eras: {mt['k95'].max() / mt['k95'].min():.2f} (pre-registered stability bar 1.5)")
mt
"""),
        md(f"""
> \U0001F4A1 **In plain words.** A pooled factor of {R['k95_pooled']}× would have worked on
> average, but the era-level factors span {R['k95_min']}–{R['k95_max']}× (ratio
> {R['k95_range']}), and the intervals are wide because a 95th percentile of 7–40 windows is
> mostly its largest one or two observations. A multiplier calibrated on 1995–2007 would have
> been blown through in 2008–2021.

## 6 · Twenty survivors

Single names, 1-year windows, 5-year lookback. **Survivor sample**: realised drawdowns are a
lower bound; breach rates a floor.
"""),
        code(r"""
S = data.load_stocks()
sc = pd.concat([st.coverage(S[c].dropna(), c, 252, (1,), 5, ("gauss", "gauss_mu0"), n_sims=2000)
                for c in S.columns], ignore_index=True)
per = sc.pivot_table(index="tape", columns="model", values="breach95", aggfunc="mean").sort_values("gauss")
fig, ax = plt.subplots(figsize=(11, 5))
per.plot.barh(ax=ax, color=["#c0392b", "#8b949e"])
ax.axvline(0.05, color="k", ls="--"); ax.set_xlabel("breach rate of the 95% band, 1-year windows")
plt.show()
print(sc.groupby("model")["breach95"].mean().round(3).to_string())
"""),
        md(f"""
> \U0001F4A1 **In plain words.** Across 540 stock-years the plug-in broke its band in
> {R['stocks_rate']} (k95 {R['stocks_k95']}, year-clustered interval {R['stocks_k95_ci']}).
> These are the names that *survived* to 2022 — the ones whose drawdown went to 100% are not
> here, so the true single-name failure is worse.

## 7 · Synthetic calibration — where the truth is known
"""),
        code(r"""
rows = []
for s in (0.0, 0.5, 1.0):
    c = st.synthetic_coverage(s, n_years=50, models=("gauss", "gauss_mu0", "gauss_oracle"),
                              n_tapes=3, n_sims=2000)
    for m, gm in c.groupby("model"):
        rows.append({"signal_strength": s, "model": m, "n": len(gm), "rate95": gm["breach95"].mean()})
sy = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9, 4.5))
for m, gm in sy.groupby("model"):
    ax.plot(gm["signal_strength"], gm["rate95"] * 100, "o-", lw=2.5, label=st.MODEL_LABELS[m])
ax.axhline(5, color="k", ls="--"); ax.set_xlabel("signal_strength (clustering + fat tails)")
ax.set_ylabel("breach rate of the 95% band, %"); ax.legend(fontsize=8); plt.show()
sy
"""),
        md(f"""
> \U0001F4A1 **In plain words.** On an i.i.d. Gaussian tape (signal_strength 0) the formula with
> the true moments breaches about 5% ({R['null_oracle']} in the full run), and estimating the
> moments from five years costs little ({R['null_plugin']}). Plant clustering and t(4) tails at
> the *same* mean and variance and the plug-in jumps to {R['planted_plugin']} — the same
> direction and size as the real tapes. The hindsight Gaussian *under*-breaches there because,
> under heavy clustering, the typical year's volatility sits well below the unconditional one.

## 8 · Robustness and limits

- **Lookback.** S&P 500 plug-in breach rates are {R['sp_lookbacks']} for 3/5/10-year and
  expanding lookbacks. On Fama-French the expanding-window plug-in falls to {R['ff_expanding']}
  — a ninety-year average mean is far less cheerful than a five-year one, which is the
  trailing-mean story again. The Nasdaq never rejects on its own (15 windows, and the dot-com
  crash precedes its first window).
- **Overlap.** The three index tapes share 1999–2018; the rule asks for two of three rather than
  pooling p-values.
- **Price indices** for S&P 500 and Nasdaq (no dividends); total return for Fama-French.

## 9 · Going further

- **Regime-switching volatility** (Hamilton-style) is the natural next challenger: GARCH
  mean-reverts too fast to know a regime, a Markov switch may not.
- **Implied volatility** as the scale (VIX is bundled for 2014–2019 only) would test whether
  the market's own forecast does better than a trailing one.
- **Conditional Drawdown-at-Risk** (Chekhlov, Uryasev & Zabarankin) as a risk budget: does the
  expected shortfall of the MDD distribution suffer the same coverage failure?
- **Contribute** a tape: the coverage test is two lines per series in `examples/verify.py`.
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
