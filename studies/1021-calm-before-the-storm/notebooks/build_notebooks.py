"""Notebook builder for Study 1021 — Stability Breeds Instability.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks walk the same seven desk beats (METHODOLOGY.md) at two altitudes: 01 is the
plain-language layer, 02 carries the inference, the confounds and the costs. The verdict
badges and every prose number come from ONE dict, ``REAL`` below, which mirrors
``docs/results.md`` (regenerate it with ``examples/verify.py`` and update the dict). The code
cells recompute from the frozen tapes, so a reader can check every number against the prose.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# The real run, pinned — mirrors docs/results.md (as-of 2018-11-30, fp 394be76d78ac)
# --------------------------------------------------------------------------- #
REAL = {
    "signal": "Weak", "trad": "Mirage",
    "as_of": "2018-11-30", "fingerprint": "394be76d78ac",
    "primary_slope": "+1.4%", "primary_t": "0.88", "primary_t_no": "0.75",
    "n_indep": "37", "n_episodes": "7",
    "null_mean": "−1.3%", "p_garch": "0.08", "p_figarch": "0.06", "p_garch_vol": "0.07",
    "crash_calm": "23.7%", "crash_rest": "20.8%",
    "ar1_phi": "0.69", "ar1_t": "14.6",
    "half1": "−0.9%", "half2": "+3.9%",
    "grid_p10": "4 of 9", "grid_t2": "0 of 9",
    "derisk_sharpe": "0.487", "bh_sharpe": "0.496", "vt_sharpe": "0.455",
    "derisk_diff": "−0.008", "derisk_p": "0.60",
    "derisk_dd": "−46%", "bh_dd": "−50%", "derisk_term": "717×", "bh_term": "986×",
    "bt_years": "67", "share_derisked": "16%",
    "vix17": "11.1", "vix18": "37.3", "spx18_dd": "−19.8%",
}

_COL = {"Real": "2ea44f", "Investable": "2ea44f", "Weak": "dab617", "Mixed": "dab617",
        "Fragile": "dab617", "None": "c0392b", "Mirage": "c0392b"}


def _badges():
    s, t = REAL["signal"], REAL["trad"]
    return (f"![Signal: {s}](https://img.shields.io/badge/Signal-{s}-{_COL[s]}"
            f"?style=flat-square)\n"
            f"![Tradability: {t}](https://img.shields.io/badge/Tradability-{t}-{_COL[t]}"
            f"?style=flat-square)")


BOOT = """\
import sys, os, warnings
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 80
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
warnings.filterwarnings("ignore")
from calm import data, strategy as st
ff = data.load_ff()
r, rf = ff["mkt"], ff["rf"]
sig = st.calm_signal(r)
print(f"Fama-French monthly TOTAL return, {ff.index[0].date()} -> {ff.index[-1].date()} "
      f"(as-of {data.AS_OF}), fingerprint {data.fingerprint(ff)}")
"""


def md(text):
    return new_markdown_cell(text.strip("\n").rstrip())


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


def F(text):
    """Fill {key} placeholders from REAL (literal braces in prose are doubled)."""
    return text.format(**REAL)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(F(r"""
# Stability Breeds Instability 🌋
### Does a long, quiet market plant the seeds of the next crash?

""" + _badges() + r"""

**Beat 0 · Verdict (real tape).** Long stretches of unusually low volatility *were* followed by
somewhat deeper drawdowns over the next two years — the opposite of what a quiet market
"should" do — but not by enough to rule out luck (a no-feedback volatility model produces a
slope at least this large {p_garch}–{p_figarch} of the time, depending on the model). And
selling the calm did not pay: a rule that halved its stocks during long calms finished with a
slightly *lower* risk-adjusted return than simply holding.

> 📓 **This is the plain-language layer.** The regressions, both simulated null worlds, the
> robustness grid and the costs are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
> Headline numbers come from the pinned run in [`docs/results.md`](../docs/results.md)
> (as-of {as_of}, fingerprint `{fingerprint}`); every chart below is recomputed from the
> frozen tapes by the code beside it.
>
> ⚠️ **Not investment advice.**
""")),
        code(BOOT),
        md(F(r"""
## 1 · The claim

The economist **Hyman Minsky** had a famous one-liner: *stability is destabilising.* When markets
are calm for years, everyone's risk models say risk is low, so everyone takes more of it —
more leverage, more selling of insurance, more reaching for yield. The calm itself builds the
fragility that ends it.

In 2018 Jón Danielsson, Marcela Valenzuela and Ilknur Zer took this to the data (*"Learning
from History: Volatility and Financial Crises"*, Review of Financial Studies). Across many
countries and two centuries, long periods of **unusually low** stock-market volatility —
measured against its own slow-moving trend — came before banking crises. On trading desks the
same idea travels as a slogan: **"low VIX = complacency = sell."**
""")),
        md(r"""
## 2 · So what?

If it's true, two things follow. First, the calm periods everyone enjoys are the *dangerous*
ones, and a regulator or a risk manager should get *more* nervous when the dashboards go quiet.
Second, there's money in it: step out of the market when calm has gone on too long and you
dodge the crash that follows.

There's a catch that makes this much harder to test than it sounds, and it's the heart of the
study.
"""),
        md(r"""
## 3 · How we'd know — and the catch

**The catch: volatility comes back.** Markets have moods. A quiet month is usually followed by
another quiet month (volatility *clusters*), but no mood lasts forever — volatility drifts
back towards its average (it *mean-reverts*). So *any* very calm stretch will, eventually, be
followed by something less calm. That happens even in a world where nobody ever changes their
behaviour.

So "calm was followed by storm" isn't enough. We need: **was it followed by *more* storm than
mean reversion alone would produce?**

The plan, fixed before looking:

1. **A "prolonged calm" score**, using only the past: how far below its own 10-year average
   volatility has been, averaged over the last 5 years.
2. **What came next**: the worst peak-to-trough fall over the next 2 years (and 1, 3, 12, 36
   months, for context).
3. **Two fake worlds** fitted to the real market — they reproduce its clustering and its
   mean reversion but contain **no** Minsky feedback. We run the identical test on hundreds of
   them. The real market has to beat them.
4. **A trade**: halve the stocks after a long calm, against simply holding.

*Mirage* would mean: the real link is no stronger than the fake worlds produce, or the trade
doesn't beat holding.
"""),
        md(r"""
## 4 · The teardown

### 4.1 Ninety years of calm and storm
"""),
        code(r"""
wealth = (1 + r).cumprod()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
a1.semilogy(wealth.index, wealth, color="#1f6feb", lw=1.4)
a1.set_ylabel("$1 grows to... (log scale)")
a1.set_title("US stock market, total return, 1926-2018", fontsize=11)
c = sig["calm"]
hi = c > c.quantile(0.75)
for ax in (a1, a2):
    ax.fill_between(c.index, 0, 1, where=hi.to_numpy(), transform=ax.get_xaxis_transform(),
                    color="#dab617", alpha=0.25, lw=0)
a2.plot(c.index, c, color="#8a6d00", lw=1.2)
a2.set_ylabel("prolonged-calm score")
a2.set_xlabel("shaded = top quarter of calm")
plt.tight_layout(); plt.show()
"""),
        md(r"""
The shaded stretches are the calmest quarter of the record: the 1940s (calm compared with
the Depression-era norm), 1994-99, 2006-2011 and 2017-18. The score is slow on purpose — it is
a five-year average — so it stays high right through the 2008 crash it "preceded". Three of the
four calm runs were followed by a fall of 20% or more (1946-47, 2000-02, 2008); the last one
runs out of data in November 2018, a month before the Christmas sell-off. That looks
impressive, and it is also only four episodes.
"""),
        md(r"""
### 4.2 Were calm periods followed by more crashes?
"""),
        code(r"""
rows = []
for H in (12, 24, 36):
    o = st.crash_odds(r, rf, H)
    rows.append({"horizon": f"{H} months", "after long calm": o["p_calm"],
                 "otherwise": o["p_rest"]})
t = pd.DataFrame(rows).set_index("horizon")
ax = (t * 100).plot.bar(color=["#dab617", "#8b949e"], figsize=(9, 4.5), rot=0)
ax.set_ylabel("% of months followed by a fall of 20%+")
plt.show()
print((t * 100).round(1).to_string())
print(f"\nseparate 20%+ crashes in the whole sample: {st.crash_odds(r, rf, 24)['n_episodes']}")
"""),
        md(F(r"""
A little more often — {crash_calm} vs {crash_rest} at two years — but these months overlap
heavily, and the whole comparison rests on only **{n_episodes} separate crashes** since the score
begins in 1942. That is very little to go on.
""")),
        md(r"""
### 4.3 The real test: beat a world with no Minsky in it
"""),
        code(r"""
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = st.mean_reversion_null(ff, "garch", 24, n_paths=300, seed=1021)
null = m["_null_samples"]["mdd"]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.hist(null * 100, bins=40, color="#8b949e", alpha=0.8,
        label="300 simulated markets with NO risk-taking feedback")
ax.axvline(m["mdd"]["real"] * 100, color="#c0392b", lw=3, label="the real market")
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_xlabel("how much deeper the next 2-year drawdown is per unit of calm (% points)")
ax.legend(fontsize=9); plt.show()
print(f"real: {m['mdd']['real']:+.2%}   fake worlds, on average: {m['mdd']['null_mean']:+.2%}")
print(f"share of fake worlds at least as extreme as the real one: {m['mdd']['p_value']:.2f}")
"""),
        md(F(r"""
This is the picture to remember. In the fake worlds — which copy the real market's mood swings
but contain no Minsky — calm is followed by *smaller* drawdowns on average ({null_mean} per
unit of calm): quiet markets tend to stay quiet for a while. The real market went the other
way ({primary_slope}). That's interesting! But roughly one fake world in thirteen to one in
eighteen does it too, by pure chance. The usual bar is one in twenty, and we don't clear it.

> 🔬 **For the quants.** One-sided p = {p_garch} against GARCH(1,1)-t and {p_figarch} against
> a long-memory FIGARCH-t, each fitted to the same 92 years. The raw Newey-West t is only
> {primary_t}. Details and a nine-way robustness grid in notebook 02.
""")),
        md(r"""
### 4.4 Two horizons, two answers
"""),
        code(r"""
prof_now = st.vol_horizon_profile(r, -sig["dev"])
prof_calm = st.vol_horizon_profile(r, sig["calm"])
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(prof_now["h"], prof_now["corr"], "o-", color="#1f6feb",
        label="vol is low RIGHT NOW")
ax.plot(prof_calm["h"], prof_calm["corr"], "s-", color="#c0392b",
        label="vol has been low for YEARS (prolonged calm)")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("months ahead"); ax.set_ylabel("link with future volatility (correlation)")
ax.legend(fontsize=9); plt.show()
"""),
        md(r"""
The blue line is the part everybody agrees on: if the market is quiet *now*, it's very likely
to be quiet over the next few months. Calm predicts calm. The red line is the Minsky claim:
after *years* of calm, volatility a year or three out tends to be a bit higher. Both are
visible; only the first is beyond doubt.
"""),
        md(F(r"""
## 5 · The verdict

**Signal: {signal}.** The real market leans Minsky's way — long calm, then somewhat deeper
falls, when a no-feedback world would predict shallower ones — but the lean sits at
p ≈ {p_garch}/{p_figarch} against two fake worlds, the raw statistic is far from significant
(t = {primary_t}), it depends on how you define "calm" (it clears p < 0.10 for {grid_p10}
reasonable definitions), and it lives entirely in the second half of the sample
({half1} before 1979, {half2} after). Literature support plus a suggestive tape reads Weak.

**Tradability: {trad}.** See below.
""")),
        md(r"""
## 6 · Could you trade it?
"""),
        code(r"""
W = {k: st.rule_weights(r, k) for k in ("buy_hold", "calm_derisk", "vol_target")}
start = max(W["calm_derisk"].first_valid_index(), W["vol_target"].first_valid_index())
names = {"buy_hold": "just hold", "calm_derisk": "halve stocks after long calm",
         "vol_target": "cut stocks when vol is high"}
cols = {"buy_hold": "#1f6feb", "calm_derisk": "#c0392b", "vol_target": "#8b949e"}
fig, ax = plt.subplots(figsize=(10, 5))
for k, w in W.items():
    bt = st.backtest(ff.loc[start:], w.loc[start:], 10.0)
    s = st.summary(bt)
    ax.semilogy((1 + bt["net"]).cumprod(), color=cols[k], lw=1.5,
                label=f"{names[k]}: Sharpe {s['sharpe_net']:.2f}, worst fall {s['max_dd']:.0%}")
ax.set_ylabel("$1 grows to... (log, after costs)"); ax.legend(fontsize=9); plt.show()
"""),
        md(F(r"""
Halving your stocks whenever calm has gone on unusually long (decided at month-end, acted on
the next month, after trading costs, with the cash earning T-bill interest) trimmed the worst
fall from {bh_dd} to {derisk_dd} — and cost you money: $1 became {derisk_term} instead of
{bh_term} over {bt_years} years, and the risk-adjusted return slipped from {bh_sharpe} to
{derisk_sharpe}. Costs aren't the problem — it barely trades. The problem is that long calms
are usually **good** times to own stocks, and the rule sits half out of them for years. The
crashes it partly dodges don't pay for the rallies it misses. **Mirage.**
""")),
        md(F(r"""
## 7 · Going further

- **The VIX anecdote.** The VIX averaged {vix17} through 2017 — the calmest year it had ever
  recorded — then spiked to {vix18} in February 2018, and the S&P 500 fell {spx18_dd} by
  Christmas. It's the story everyone tells, and it's **one** episode on a five-year tape. One
  story can't prove a pattern; notebook 02 shows it, labelled as exactly that.
- **Crises are not crashes.** Danielsson and co-authors predict *banking crises* across 60
  countries. We tested the trader's version on US stocks. A fork with a cross-country crisis
  dataset would test the original claim.
- **The best lead we left open.** Long calm lines up with *higher volatility* one to three years
  out more clearly than with crashes. It wasn't our pre-registered test, so it doesn't move the
  verdict — but it's the first thing to pre-register next.
- **What to challenge:** the calm definition (10-year trend, 5-year window), the 2-year horizon,
  the fake-world models. Each is one argument in [`calm/strategy.py`](../calm/strategy.py).
""")),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(F(r"""
# Stability Breeds Instability — a quantitative teardown 🔬
### Prolonged low vol vs forward drawdowns · HAC and non-overlapping · GARCH and FIGARCH mean-reversion nulls

""" + _badges() + r"""

The companion to the [notebook for the curious](01_for_the_curious.ipynb). The study turns on
**§4**: the real slope of forward drawdown on prolonged calm, set against the distribution
the same regression produces on tapes simulated from no-feedback volatility models fitted to
the same 92 years. Volatility mean reversion is the confound, and in those models it pushes
the slope **negative** — so the bar is not zero, it is the null.

> **Beat 0 · Verdict (real tape)** — numbers sourced to [`docs/results.md`](../docs/results.md),
> as-of {as_of}, fingerprint `{fingerprint}`. The executed cells recompute them from the
> frozen tapes; the synthetic controls in §8 are labelled as such and never feed a stamp.
>
> ⚠️ **Not investment advice.** Fama-French `Mkt-RF + RF` is a **total** return; the S&P 500
> tape (§6) is a **price** index; the VIX tape (§7) is five years and used as an anecdote.
>
> 💡 **The `💡 In plain words` notes** translate each result back into intuition.
""")),
        code(BOOT),
        md(F(r"""
## Verdict, up front

| Axis | Stamp | Decisive numbers |
|---|---|---|
| **Signal** | {signal} | primary slope {primary_slope} per SD of calm on the 24-month max drawdown, HAC t = {primary_t} (non-overlap {primary_t_no}, ~{n_indep} windows); no-feedback null mean {null_mean}; one-sided p = **{p_garch}** (GARCH-t), **{p_figarch}** (FIGARCH-t); halves {half1} / {half2}; {n_episodes} independent ≥20% drawdowns |
| **Tradability** | {trad} | calm de-risk net excess Sharpe {derisk_sharpe} vs {bh_sharpe} buy-and-hold (Δ {derisk_diff}, block-bootstrap p(Δ≤0) = {derisk_p}); vol-target {vt_sharpe}; max DD {derisk_dd} vs {bh_dd} |

> 💡 **In plain words.** The tape leans Minsky's way and a no-feedback world leans the other
> way, but the gap is a 1-in-13-to-1-in-17 event under the null, not 1-in-20 — and the trade
> loses Sharpe.

## 1 · Hypotheses (pre-registered)

- **H₁ (primary).** The slope of the 24-month forward max drawdown on the standardised
  prolonged-calm score is positive with HAC t ≥ 2.
- **H₂ (the confound).** That slope exceeds the slope produced by a GARCH(1,1)-t **and** a
  FIGARCH-t fitted to the same tape, one-sided p < 0.05 each.
- **H₃ (two horizons).** Low vol now predicts low vol next month (clustering, AR(1) > 0), while
  prolonged calm predicts *more* vol and drawdown at 1-3 years.
- **H₄ (trade).** Halving equity after prolonged calm beats buy-and-hold on net excess Sharpe
  (block-bootstrap p < 0.05) with a shallower drawdown, and beats a plain vol-target.

The verdict rule (`strategy.verdict`) encodes H₁+H₂ → Real, a raw-only or p < 0.10 result → Weak,
H₄ → Investable; it is unit-tested in both directions.
""")),
        md(r"""
## 2 · The signal

`rv_t` = std of the last 12 monthly total returns × √12; `trend_t` = trailing 120-month mean of
`rv`; `low_t = max(−log(rv_t/trend_t), 0)`; `calm_t` = trailing 60-month mean of `low`. All
past-only: `calm_t` is known at the close of month t, and the first reading needs 189 months
of history (first value 1942-04).
"""),
        code(r"""
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(sig["rv"], color="#1f6feb", lw=1, label="12m realised vol")
ax[0].plot(sig["trend"], color="k", lw=1.5, label="10-year trend (past-only)")
ax[0].legend(fontsize=9); ax[0].set_ylabel("annualised vol")
ax[1].plot(sig["calm"], color="#c0392b", lw=1.2)
ax[1].set_ylabel("calm score (5y mean of below-trend shortfall)")
plt.tight_layout(); plt.show()
print(sig["calm"].describe().round(3).to_string())
"""),
        md(r"""
## 3 · Every horizon, every outcome

Slope per 1 SD of calm, Newey-West with 2H lags; the non-overlapping t keeps every H-th month
(median across offsets).
"""),
        code(r"""
tab = st.horizon_table(r, rf)
print(tab.round(4).to_string(index=False))
piv = tab.pivot(index="outcome", columns="H", values="t_hac")
fig, ax = plt.subplots(figsize=(9, 3.6))
im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"H={h}" for h in piv.columns])
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.iat[i, j]:+.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="HAC t"); ax.set_title("HAC t of calm -> outcome", fontsize=10)
plt.show()
"""),
        md(F(r"""
> 💡 **In plain words.** Every long-horizon sign points Minsky's way and none clears 2. The
> primary cell (24-month drawdown) has t = {primary_t}; thinning to ~{n_indep} non-overlapping
> windows gives {primary_t_no} — HAC is not hiding anything.
""")),
        md(r"""
## 4 · The confound — mean reversion, simulated

Both models are fitted (constant mean, Student-t) to the real monthly excess returns, simulated
at the real length with the real bill path added back, and the identical 24-month regression is
run on every path. p is one-sided in Minsky's direction.
"""),
        code(r"""
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_g = st.fit_vol_model(ff["mkt_rf"], "garch")
    res_f = st.fit_vol_model(ff["mkt_rf"], "figarch")
    NG = st.mean_reversion_null(ff, "garch", 24, 400, seed=1021, res=res_g)
    NF = st.mean_reversion_null(ff, "figarch", 24, 400, seed=1021, res=res_f)
print("GARCH  :", {k: round(v, 3) for k, v in NG["params"].items()})
print("FIGARCH:", {k: round(v, 3) for k, v in NF["params"].items()})
rows = []
for name, N in (("GARCH-t", NG), ("FIGARCH-t", NF)):
    for o in st.OUTCOMES:
        q = N[o]
        rows.append({"null": name, "outcome": o, "real": q["real"], "null_mean": q["null_mean"],
                     "q05": q["null_q05"], "q95": q["null_q95"], "p": q["p_value"]})
print(pd.DataFrame(rows).round(4).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (name, N) in zip(axs, (("GARCH(1,1)-t", NG), ("FIGARCH-t", NF))):
    x = N["_null_samples"]["mdd"]
    x = x[np.isfinite(x)]
    ax.hist(x * 100, bins=40, color="#8b949e")
    ax.axvline(N["mdd"]["real"] * 100, color="#c0392b", lw=3, label=f"real (p={N['mdd']['p_value']:.3f})")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_title(f"{name}: null slope of 24m max DD", fontsize=10)
    ax.set_xlabel("slope per SD of calm, % points"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
"""),
        md(F(r"""
> 💡 **In plain words.** With no feedback, calm is followed by *calmer* (null mean {null_mean}):
> GARCH persistence means a quiet market stays quiet for a while. So mean reversion does not
> manufacture the Minsky slope here — it works *against* it, which makes the real positive slope
> more interesting, not less. It still lands at p = {p_garch} / {p_figarch}: the right tail, not
> past the 5% line. The FIGARCH null is wider (long memory makes the slow component wander),
> and the two agree.
""")),
        md(r"""
## 5 · Robustness — signal definition and sample halves

The pre-registered point is (120, 60). The grid is shown, not chosen.
"""),
        code(r"""
grid = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fo24 = st.forward_outcomes(r, rf, 24)
    for tw in (60, 120, 180):
        for cw in (36, 60, 84):
            win = {"trend_window": tw, "calm_window": cw}
            rr = st.predictive_regression(st.calm_signal(r, **win)["calm"], fo24["mdd"], 24)
            m_ = st.mean_reversion_null(ff, "garch", 24, 200, seed=7, res=res_g, windows=win)
            grid.append({"trend": tw, "calm": cw, "t": rr["t"], "p": m_["mdd"]["p_value"]})
G = pd.DataFrame(grid)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, col, cmap, lim in ((axs[0], "t", "RdBu_r", (-2.5, 2.5)), (axs[1], "p", "viridis_r", (0, 0.5))):
    P = G.pivot(index="trend", columns="calm", values=col)
    im = ax.imshow(P.to_numpy(), cmap=cmap, vmin=lim[0], vmax=lim[1])
    ax.set_xticks(range(3)); ax.set_xticklabels([f"calm {c}m" for c in P.columns])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"trend {t}m" for t in P.index])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{P.iat[i, j]:.2f}", ha="center", va="center", color="w")
    ax.set_title("HAC t" if col == "t" else "GARCH null p", fontsize=10)
plt.tight_layout(); plt.show()
for x in st.half_sample_slopes(r, rf):
    print(f"half {x['start']} -> {x['end']}: slope {x['slope']:+.4f}  HAC t {x['t']:+.2f}")
"""),
        md(F(r"""
> 💡 **In plain words.** Measure "calm" against a 5-year norm and the case strengthens
> (p ≈ 0.01); against a 15-year norm it vanishes, even flips. The registered 10-year version sits
> in between. And the whole effect lives after 1979 ({half2} per SD) — before that it points the
> other way ({half1}). A result that depends this much on the ruler and the decade is the
> definition of Weak.
""")),
        md(r"""
## 6 · Two horizons — clustering vs the Minsky claim

The short-horizon sign on the daily S&P 500 (price index; monthly realised vol from daily
returns), then the full horizon profile on Fama-French with a GARCH null band.
"""),
        code(r"""
spx = data.load_spx_daily()
rvm = data.monthly_rv_from_daily(spx)
ar = st.clustering_ar1(rvm)
print(f"S&P 500 1990-2022, log monthly RV AR(1): phi = {ar['phi']:.3f} (HAC t {ar['t']:.1f}), "
      f"half-life {ar['half_life_months']:.1f} months, n = {ar['n']}")
prof_now = st.vol_horizon_profile(r, -sig["dev"])
prof_calm = st.vol_horizon_profile(r, sig["calm"])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    band = st.null_horizon_profile(res_g, "garch", ff, n_paths=60)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
y = np.log(rvm)
axs[0].scatter(y.shift(1), y, s=6, alpha=0.5, color="#1f6feb")
axs[0].set_xlabel("log RV, this month"); axs[0].set_ylabel("log RV, next month")
axs[0].set_title(f"clustering: phi = {ar['phi']:.2f}", fontsize=10)
axs[1].fill_between(band.index, band["q05"], band["q95"], color="#8b949e", alpha=0.3,
                    label="GARCH no-feedback 90% band (calm)")
axs[1].plot(band.index, band["mean"], color="#8b949e", lw=1)
axs[1].plot(prof_now["h"], prof_now["corr"], "o-", color="#1f6feb", ms=3, label="low vol now")
axs[1].plot(prof_calm["h"], prof_calm["corr"], "s-", color="#c0392b", ms=3, label="prolonged calm")
axs[1].axhline(0, color="k", lw=1); axs[1].set_xlabel("months ahead (h)")
axs[1].set_ylabel("corr with log vol over t+h+1..t+h+6"); axs[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
        md(F(r"""
> 💡 **In plain words.** Short horizon: calm now → calm next month, φ = {ar1_phi}, t = {ar1_t}.
> No debate. Long horizon: prolonged calm → *more* vol one to three years out, and the red curve
> pokes above the null's pointwise band for a long stretch. It is the strongest-looking picture
> in the study — and it is a post-hoc scan of 21 overlapping horizons against a 60-path band,
> so it moves the stamp nowhere. Pre-registering "calm → 24-month forward vol, correlation
> null" is the obvious fork (beat 7).
""")),
        md(r"""
## 7 · Illustrations — not inference

The post-1990 S&P tape only produces a calm score from late 2005; the VIX tape is five years.
"""),
        code(r"""
mret = data.monthly_returns_from_daily(spx)
s3 = st.calm_signal(mret)["calm"]
fo3 = st.forward_outcomes(mret, None, 24)
print(pd.DataFrame({"calm": s3, "next 24m max DD (price)": fo3["mdd"]}).loc[
    ["2006-12-31", "2007-06-30", "2014-06-30", "2017-12-31", "2019-12-31"]].round(3).to_string())
vix = data.load_vix()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(vix.index, vix, color="#c0392b", lw=1, label="VIX")
ax2 = ax.twinx()
s = spx.loc[vix.index[0]:vix.index[-1]]
ax2.plot(s.index, s, color="#1f6feb", lw=1, label="S&P 500 (price)")
ax.set_ylabel("VIX"); ax2.set_ylabel("S&P 500")
ax.set_title("2014-2018: the 2017 calm, then 2018. One episode — an anecdote, not evidence",
             fontsize=10)
ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="upper center", fontsize=9)
ax2.grid(False); plt.show()
v17 = vix[vix.index.year == 2017]
print(f"VIX 2017 mean {v17.mean():.1f}, below 11 on {(v17 < 11).mean():.0%} of days; "
      f"2018 max {vix[vix.index.year == 2018].max():.1f}")
"""),
        md(r"""
> 💡 **In plain words.** 2007 and 2017 are the two calm peaks every believer cites, and both were
> followed by a fall. They are two observations. The 92-year tape in §3-§5 is where the
> inference lives.
"""),
        md(r"""
## 8 · Machinery proof — the synthetic controls (not market evidence)

Two synthetic worlds from `data.synthetic_monthly`: same GARCH, same crash rate; in one a
crash is likelier after prolonged calm (`signal_strength=1`), in the other crashes are blind to
it (`0`). The harness must fire on the first and stay quiet on the second.
"""),
        code(r"""
rows = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for s_ in (1.0, 0.0):
        f, tr = data.synthetic_monthly(signal_strength=s_, seed=1021)
        c_ = st.calm_signal(f["mkt"])["calm"]
        fo_ = st.forward_outcomes(f["mkt"], f["rf"], 24)
        rr = st.predictive_regression(c_, fo_["mdd"], 24)
        m_ = st.mean_reversion_null(f, "garch", 24, 200, seed=5)
        rows.append({"world": "planted" if s_ else "null", "crashes": tr["n_crashes"],
                     "slope": rr["slope"], "HAC t": rr["t"],
                     "GARCH null mean": m_["mdd"]["null_mean"], "p": m_["mdd"]["p_value"]})
print("SYNTHETIC — machinery proof only")
print(pd.DataFrame(rows).round(4).to_string(index=False))
"""),
        md(r"""
> 💡 **In plain words.** When the feedback is there, the harness finds it (p well under 0.05);
> when it isn't, the null comparison says so, even though both worlds have the same number of
> crashes. So the real tape's p ≈ 0.06-0.08 is a statement about the market, not a blind spot
> in the method.
"""),
        md(r"""
## 9 · Could you trade it?

Weights decided at the close of month t, held through t+1 (one shift); turnover one-way × NAV
from the drifted weight; 10 bp one-way; cash earns bills; Sharpe on excess-of-bills for every leg.
"""),
        code(r"""
W = {k: st.rule_weights(r, k) for k in ("buy_hold", "calm_derisk", "vol_target")}
start = max(W["calm_derisk"].first_valid_index(), W["vol_target"].first_valid_index())
ffb = ff.loc[start:]
BT = {k: st.backtest(ffb, w.loc[start:], 10.0) for k, w in W.items()}
S = pd.DataFrame({k: st.summary(b) for k, b in BT.items()}).T
print(S[["sharpe_gross", "sharpe_net", "cagr_net", "vol", "max_dd", "terminal",
         "turnover_yr", "avg_weight"]].astype(float).round(3).to_string())
ex = {k: b["net"] - b["rf"] for k, b in BT.items()}
for k in ("calm_derisk", "vol_target"):
    b = st.sharpe_diff_bootstrap(ex[k], ex["buy_hold"])
    print(f"{k:12s} - buy&hold: dSharpe {b['diff']:+.3f}  95% CI [{b['ci'][0]:+.3f}, "
          f"{b['ci'][1]:+.3f}]  p(<=0) {b['p_le_zero']:.2f}")
print(st.cost_sweep(ffb, W["calm_derisk"].loc[start:]).round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 4.5))
for k, col in (("buy_hold", "#1f6feb"), ("calm_derisk", "#c0392b"), ("vol_target", "#8b949e")):
    w_ = (1 + BT[k]["net"]).cumprod()
    ax.plot(w_ / w_.cummax() - 1, color=col, lw=1, label=k)
ax.set_ylabel("drawdown (net)"); ax.legend(fontsize=9); plt.show()
w = BT["calm_derisk"]["w"]
mex = (ffb["mkt"] - ffb["rf"]).reindex(w.index)
print(f"de-risked {(w < 1).mean():.0%} of months; market excess then {mex[w < 1].mean()*12:.1%}/yr, "
      f"otherwise {mex[w >= 1].mean()*12:.1%}/yr")
"""),
        md(F(r"""
> 💡 **In plain words.** The rule trades about once every twenty years' worth of NAV — costs are
> irrelevant. It loses on *timing*: it was de-risked {share_derisked} of the time, and the
> market earned **more** in those calm months than in the rest. Shallower drawdown
> ({derisk_dd} vs {bh_dd}), lower Sharpe ({derisk_sharpe} vs {bh_sharpe}), less money. A plain
> vol-target, which reacts to vol that is high *now*, did no better here on a monthly tape. **Mirage.**

## 10 · Going further

- **Pre-register the vol leg.** Calm → 24-month forward realised vol, scored by correlation
  against a no-feedback null, on a tape that was not used to find it (international indices,
  or daily data pre-1990).
- **Crises, not crashes.** Danielsson, Valenzuela & Zer predict banking crises across 60
  countries; joining a cross-country crisis chronology to this harness would test the claim as
  written.
- **A harsher null.** A Markov-switching volatility model has a fatter slow component than
  FIGARCH; if the real slope stops beating it, the case weakens further.
- **The ruler matters.** §5 shows the result hinges on the trend window. A fork that chose it
  out-of-sample (e.g. by expanding-window cross-validation) would remove a researcher degree of
  freedom this study only reports.
""")),
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
