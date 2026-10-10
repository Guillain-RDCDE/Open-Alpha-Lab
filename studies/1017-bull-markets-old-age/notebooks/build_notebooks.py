"""Notebook builder for Study 1017 — Do Bull Markets Die of Old Age?

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    cd notebooks
    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks walk the same seven desk beats (METHODOLOGY.md) at two altitudes: 01 is the
plain-language layer, 02 carries the inference, the nulls and the costs. The executed cells run
the real bundled tapes (offline, SHA-pinned) with fewer null paths than the pinned run, so their
printed p-values wobble in the second decimal; the **headline numbers** quoted in prose come from
the single ``REAL`` dict below, which mirrors ``docs/results.md`` (1,000 paths per null). Re-run
``examples/verify.py`` and refresh ``REAL`` together, or they will drift apart.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# The real run, as pinned in docs/results.md (examples/verify.py). One place only.
# ---------------------------------------------------------------------------
REAL = {
    "as_of": "2022-11-30", "as_of_ff": "2018-11-30", "fp_ff": "394be76d78ac",
    "fp_sp": "e223c581ae22",
    "signal": "None", "trad": "Mirage",
    "n_bulls": 11, "n_years": 92, "median_bull": 49,
    "k": 1.10, "k_lo": 0.72, "k_hi": 2.00, "p_k1": 0.70,
    "k_null_rw": 1.22, "k_null_garch": 0.91, "p_k_rw": 0.72, "p_k_garch": 0.24,
    "share_rw_k_gt_1": 0.88, "rw_naive_reject": 0.18,
    "n_bulls_null_rw": 16, "median_bull_null_rw": 35,
    "sens_min_p": 0.24,
    "pred_slope": -0.0039, "pred_t": -0.99, "pred_t_nov": -0.95, "p_pred_rw": 0.35,
    "p_pred_garch": 0.33, "dd_t": -0.69,
    "old_minus_young": 0.0084, "old_minus_young_t": 0.24,
    "sharpe_bh": 0.568, "sharpe_net": 0.489, "sharpe_diff": -0.079, "sharpe_lo": -0.165,
    "sharpe_hi": 0.006, "p_sharpe": 0.965, "tw_bh": 4787, "tw_net": 773, "tw_cm": 1079,
    "share_derisked": 0.59, "sp_sharpe_diff": -0.087,
    "one_sentence": (
        "Bull markets only look like they die of old age (Weibull k = 1.10) because a ±20% "
        "ruler builds ageing into any series — random walks dated the same way age at least as "
        "fast (p = 0.72), bull age does not forecast next year's return (t = −0.99), and selling "
        "half of every old bull ended with 6.2× less money than buy-and-hold."),
}

BADGE = {"Real": "2ea44f", "Weak": "dab617", "Mixed": "dab617", "None": "c0392b",
         "Investable": "2ea44f", "Fragile": "dab617", "Mirage": "c0392b"}


def badges() -> str:
    s, t = REAL["signal"], REAL["trad"]
    return (f"![Signal: {s}](https://img.shields.io/badge/Signal-{s}-{BADGE[s]}?style=flat-square)\n"
            f"![Tradability: {t}](https://img.shields.io/badge/Tradability-{t}-{BADGE[t]}"
            f"?style=flat-square)")


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.0)
plt.rcParams["figure.dpi"] = 80
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from oldage import data, strategy as st
"""

LOAD = """\
ff = data.load_ff_monthly()                    # Fama-French market, TOTAL return, monthly
sp = data.load_sp500_daily()                   # S&P 500 PRICE index, daily (no dividends)
lv = data.total_return_index(ff["mkt"])
print(f"FF monthly total return {ff.index[0]:%Y-%m} → {ff.index[-1]:%Y-%m}  "
      f"fingerprint {data.fingerprint(ff)}")
print(f"S&P 500 daily price     {sp.index[0]:%Y-%m-%d} → {sp.index[-1]:%Y-%m-%d}  "
      f"fingerprint {data.fingerprint(sp)}")
"""


def md(text):
    return new_markdown_cell(text.strip("\n").rstrip())


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


R = REAL


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(f"""
# Do Bull Markets Die of Old Age? \U0001F402
### "This bull market is long in the tooth" — checked against 92 years and a few thousand fake markets

{badges()}

Every long rally ends with the same headline: *the bull is getting old*. It sounds like common
sense — things that have lived a long time are closer to the end. We dated every US bull market
since 1926 and asked whether an old bull really is more likely to die next month than a young one.
It isn't — and the reason it *looks* like it is turns out to be the ruler, not the market.

> \U0001F4D3 **This is the plain-language layer.** The duration model, the two null worlds, the
> predictive regressions and the bootstraps are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> **Beat 0 · Verdict (real tape).** The badges and the numbers quoted in the text come from the
> pinned run in [`docs/results.md`](../docs/results.md) (as-of {R['as_of']}). The charts below are
> drawn live from the same bundled tapes.
>
> ⚠️ **Not investment advice.** Research and education only.
"""),
        code(BOOT),
        md(f"""
## The answer first \U0001F3AF

| Question | Answer |
|---|---|
| Do older bulls end more often? | Not more than you'd expect from chance. |
| Why does it *look* like they do? | The 20% rule can't kill a baby bull — so *any* market looks like it ages. |
| Does an old bull warn of a bad year ahead? | No: t = {R['pred_t']:+.2f}. |
| Should you sell half when the bull gets old? | It cost you **{R['tw_bh'] / R['tw_net']:.1f}×** your final wealth since 1940. |

> **In one sentence:** {R['one_sentence']}
"""),
        md("""
## 1 · The claim

"Long in the tooth", "late cycle", "on borrowed time". The belief has a precise shape: the
**older** a bull market is, the **more likely** it is to end soon. If that's true, the age of the
current bull is a free risk signal — and a sensible investor trims stocks once a bull has outlived
the typical one.

It's the same idea economists once had about business expansions ("they die of old age"), and
which they tested, carefully, decades ago. We run that test on the stock market.
"""),
        code(LOAD),
        code("""
cy = st.date_cycles(lv, 0.20)
fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(lv.index, lv, color="k", lw=1)
for r in cy[cy.phase == "bull"].itertuples():
    ax.axvspan(r.start, r.end, color="#2ea44f", alpha=0.18, lw=0)
ax.set_title("US stock market, total return since 1926 — green = bull markets (±20% rule)")
ax.set_ylabel("$1 grows to… (log scale)")
plt.show()
b = cy[cy.phase == "bull"]
print(b[["start", "end", "duration", "amplitude", "censored"]]
      .assign(start=lambda d: d.start.dt.strftime("%Y-%m"), end=lambda d: d.end.dt.strftime("%Y-%m"))
      .rename(columns={"duration": "months", "amplitude": "gain"}).to_string(index=False))
"""),
        md(f"""
That's the entire sample: **{R['n_bulls']} completed bull markets** in {R['n_years']} years (plus
the one still running when the data stop in 2018). Every number below rests on those few.
Twelve is not a lot — keep that in mind; we'll come back to it.

## 2 · So what?

If old bulls really were fragile, you would want to own fewer stocks late in a long rally and more
early on — a simple, cheap rule that, if it worked, would improve almost anyone's portfolio. And it
would mean markets have a *clock* in them: that the passage of time itself, not news, wears a rally
out.

## 3 · How we'd know

A rough way to check: line up all bull markets by age and ask, at each age, what share of the bulls
still alive ended within the next year. If old bulls are fragile, that share should climb with age.

**The catch we set up in advance:** the 20% rule itself bends that curve. A bull can't even be
*called* a bull until prices are 20% off the bottom, and can't end until they're 20% off the top.
So very young bulls almost never "die", and any series dated this way — even a coin-flipping random
walk — will look like it ages. The fair comparison is therefore not *"is the curve rising?"* but
*"is it rising **faster** than for fake markets with no memory at all?"* If real bulls age like fake
ones, the claim is a **mirage of the ruler**.
"""),
        md("""
## 4 · The teardown

### 4.1 · How long do bulls live?
"""),
        code("""
d, c = st.bull_durations(lv, 0.20)
fig, ax = plt.subplots(figsize=(10, 4))
order = np.argsort(d)
ax.barh(range(len(d)), d[order] / 12, color=["#8b949e" if cc else "#2ea44f" for cc in c[order]])
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"bull #{i+1}" for i in order], fontsize=8)
ax.set_xlabel("age at death, years (grey = still alive when the data stop)")
plt.show()
print(f"median completed bull: {np.median(d[~c]):.0f} months; shortest {d.min():.0f}, longest {d.max():.0f}")
"""),
        md("""
Some bulls last two months; some last fourteen years. Wide spread, few data points.

### 4.2 · The ruler trick, shown

Here is a market we *made up*: a pure random walk with the same average return and the same
month-to-month swings as the real thing. It has no memory, by construction — last month tells you
nothing about next month, and certainly nothing about "age". We date it with the same 20% rule.
"""),
        code("""
lr = np.log1p(ff["mkt"])
fake = st.simulate_log_returns("rw", len(lr), 1, {"mu": lr.mean(), "sigma": lr.std()}, seed=7)[0]
fake_lv = pd.Series(np.exp(np.cumsum(fake)), index=ff.index)
fcy = st.date_cycles(fake_lv, 0.20)
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, s, cc, ttl in ((axes[0], lv, cy, "REAL market"), (axes[1], fake_lv, fcy, "FAKE random walk (no memory)")):
    ax.semilogy(s.index, s, color="k", lw=1)
    for r in cc[cc.phase == "bull"].itertuples():
        ax.axvspan(r.start, r.end, color="#2ea44f", alpha=0.18, lw=0)
    ax.set_title(ttl)
plt.tight_layout(); plt.show()
"""),
        md("""
Could you tell which one is real without the labels? The fake one has bull markets and bear
markets too — long ones, short ones, crashes. The "cycle" is something the dating rule finds in
any wiggly line that drifts up.
"""),
        code("""
res = st.duration_test(ff["mkt"], 0.20, n_sims=200, n_boot=300, rf=ff["rf"])
real_lt = res["life"]["annual_hazard"]
rw_lt = res["nulls"]["rw"]["life"]["annual_hazard"]
x = np.arange(len(real_lt))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - 0.2, real_lt * 100, 0.4, color="#1f6feb", label="real bulls")
ax.bar(x + 0.2, rw_lt * 100, 0.4, color="#8b949e", label="random-walk bulls (no memory)")
ax.set_xticks(x); ax.set_xticklabels(real_lt.index)
ax.set_xlabel("age of the bull"); ax.set_ylabel("% chance it ends within a year")
ax.legend(); plt.show()
"""),
        md(f"""
The blue bars are what believers point at: real bulls ended **8%** of the time in their
first year and about twice that by year three. But look at the grey bars — the memoryless fake
market *also* shows a low first year, because the rule can't kill a baby bull. After year three
the real hazard doesn't keep climbing; it flattens around 13–15% a year. **Old bulls are not
fragile; young bulls are just hard to date.**

> \U0001F52C **For the quants.** A Weibull duration model gives the real bulls a shape
> k = {R['k']:.2f}; random walks dated the same way give a median k = {R['k_null_rw']:.2f}
> ({R['share_rw_k_gt_1']:.0%} of them above 1). One-sided p = {R['p_k_rw']:.2f} against the random
> walk, {R['p_k_garch']:.2f} against a GARCH world. A naive test of "k = 1" calls a memoryless
> random walk "dies of old age" {R['rw_naive_reject']:.0%} of the time.

### 4.3 · Does an old bull warn of a bad year?
"""),
        code("""
rt = st.realtime_state(lv, 0.20)          # what you could have known at each month-end
fwd = st.forward_log_return(ff["mkt"], ff["rf"], 12)
m = (rt.state == 1) & rt.age.notna() & fwd.notna()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(rt.age[m] / 12, fwd[m] * 100, s=6, alpha=0.35, color="#1f6feb")
pr = st.predictive_regression(ff["mkt"], ff["rf"])
xs = np.linspace(0, (rt.age[m] / 12).max(), 50)
ax.plot(xs, (pr["alpha"] + pr["slope"] * xs) * 100, color="#c0392b", lw=2.5, label="best-fit line")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("age of the bull at month-end (years)"); ax.set_ylabel("next 12 months, % over T-bills")
ax.legend(); plt.show()
print(f"slope: {pr['slope']:+.2%} per year of age, t = {pr['t']:+.2f}")
"""),
        md(f"""
A cloud with a line through it that is very nearly flat. A bull that is ten years old has been
followed by roughly the same next year as a bull that is two.

## 5 · The verdict

**Signal: {R['signal']}.** Real bull markets age no faster than the bulls of a random walk dated
with the same ruler (p = {R['p_k_rw']:.2f}), and a bull's age does not predict the next year's
return (t = {R['pred_t']:+.2f}) or drawdown. The "long in the tooth" pattern is real *in the chart* —
it's produced by how we draw the chart.

## 6 · Could you trade it?

The believers' rule: hold 100% stocks, drop to 50% (rest in T-bills) once the current bull is older
than the typical one — judged only on what was known at the time, acting a month later, paying
costs.
"""),
        code("""
W = st.age_rule_weights(lv, 0.20)
start = W.median_age.first_valid_index()
bt = st.age_rule_backtest(ff["mkt"], ff["rf"], W.weight, 10.0)
bt = bt[bt.index > start]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.semilogy((1 + bt.bh).cumprod(), color="k", lw=1.6, label="buy and hold")
ax.semilogy((1 + bt.net).cumprod(), color="#c0392b", lw=1.6, label="sell half when the bull is 'old'")
ax.set_ylabel("$1 grows to… (log)"); ax.legend(); plt.show()
s = st.summarize_backtest(bt)
print(f"since {start:%Y-%m}: $1 → ${s['tw_bh']:,.0f} buy-and-hold vs ${s['tw_net']:,.0f} with the rule")
"""),
        md(f"""
**Tradability: {R['trad']}.** The rule spent {R['share_derisked']:.0%} of the months half in cash,
mostly during long bulls that kept going, and was fully invested exactly when the market was in
its rough patches. Even per unit of risk it lost (Sharpe {R['sharpe_net']:.2f} vs
{R['sharpe_bh']:.2f}), and it also lost on the separate S&P 500 daily tape since 1990.

## 7 · Going further \U0001F6AA

- **Twelve bulls is the hard ceiling.** No method squeezes certainty out of a dozen events — the
  honest conclusion is "no evidence of ageing beyond the ruler", not "proven memoryless".
- **What *might* age is valuation, not time.** Long bulls tend to end expensive; whether *price*
  rather than *age* predicts the end is a different study (and the desk has several on valuation).
- **Try other rulers.** The quants notebook reruns everything at 15%, 25% and an asymmetric rule.
  A business-cycle-style dating algorithm (Pagan–Sossounov) is a natural fork.
- **The general lesson:** whenever a pattern is found by a rule that *defines* the events, check
  what the rule finds in data that has no pattern at all.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(f"""
# Do Bull Markets Die of Old Age? — a quantitative teardown \U0001F52C
### Censored Weibull hazards · random-walk and GARCH nulls through the same dating rule · predictive regressions · a de-risk rule

{badges()}

The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The comparison that
carries the study is **§4.3**: the Weibull shape of real bull markets judged not against k = 1 but
against the distribution of k for random-walk and GARCH(1,1)-t tapes with the same drift and
volatility, dated by the **same** ±20% rule. **§4.6** asks whether the real-time age of a bull
predicts the next 12 months, with Newey-West errors *and* a simulated null for the persistent
regressor; **§6** books the de-risking rule.

> **Beat 0 · Verdict (real tape).** Badges and prose numbers are sourced to
> [`docs/results.md`](../docs/results.md) (as-of {R['as_of']}, FF fingerprint `{R['fp_ff']}`,
> S&P fingerprint `{R['fp_sp']}`, 1,000 paths per null). Cells below re-run the real tapes with 200
> null paths, so their p-values move in the second decimal.
>
> ⚠️ **Not investment advice.** The FF market series is **total return**; the S&P 500 tape is a
> **price index** (no dividends).
>
> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT),
        code(LOAD),
        md(f"""
## Verdict, up front

| Axis | Stamp | Decisive numbers |
|---|---|---|
| **Signal** | {R['signal']} | Real k = {R['k']:.2f} (95% CI {R['k_lo']:.2f}–{R['k_hi']:.2f}) vs random-walk median {R['k_null_rw']:.2f} (p = {R['p_k_rw']:.2f}) and GARCH median {R['k_null_garch']:.2f} (p = {R['p_k_garch']:.2f}); predictive slope NW t = {R['pred_t']:+.2f}, simulated p = {R['p_pred_rw']:.2f}; smallest p across four dating rules {R['sens_min_p']:.2f}. |
| **Tradability** | {R['trad']} | De-risk rule net Sharpe {R['sharpe_net']:.3f} vs {R['sharpe_bh']:.3f} (Δ {R['sharpe_diff']:+.3f}, CI {R['sharpe_lo']:+.3f} to {R['sharpe_hi']:+.3f}); $1 → ${R['tw_net']:,} vs ${R['tw_bh']:,}; S&P daily Δ {R['sp_sharpe_diff']:+.3f}. |

> \U0001F4A1 **In plain words.** Real bulls age exactly like bulls of a market with no memory, once
> both are measured with the same ruler — and selling old bulls just forfeits the equity premium.

## 1 · Hypotheses (fixed before the run)

- **H₁ (excess ageing).** The Weibull shape of real bull durations exceeds the 95th percentile of
  the shape produced by a random walk *and* by a GARCH(1,1)-t with the tape's drift, dated by the
  same rule. (`verdict`: Real iff the larger of the two one-sided p-values < 0.05.)
- **H₂ (prediction).** Real-time bull age has a negative slope for the next-12-month excess log
  return, NW t ≤ −2 and simulated p < 0.05 against both nulls.
- **H₃ (tradability).** Cutting equity to 50% in bulls older than the causal median improves the
  net excess-of-cash Sharpe (block-bootstrap p < 0.05; Fragile if p < 0.20), and the S&P daily tape
  agrees.
- **Null.** A raw k > 1, or a bootstrap CI excluding 1, earns **nothing** on its own.

## 2 · The dating rule
"""),
        code("""
cy = st.date_cycles(lv, 0.20)
print(cy.assign(start=cy.start.dt.strftime("%Y-%m"), end=cy.end.dt.strftime("%Y-%m"),
                confirmed=cy.confirmed.dt.strftime("%Y-%m"))
        [["phase", "start", "end", "duration", "amplitude", "confirmed", "censored"]].to_string(index=False))
b = cy[cy.phase == "bull"]
lag = ((b.confirmed - b.end).dt.days / 30.44).dropna()
print(f"\\nmedian months from the peak to its confirmation: {lag.median():.1f}")
"""),
        md("""
> \U0001F4A1 **In plain words.** A top is only known once prices are 20% below it — about half a
> year later, typically. That's why the trading rule in §6 has to use the *real-time* clock.

The S&P 500 daily price index, same rule — only a handful of cycles, so it serves as a cross-check:
"""),
        code("""
spc = st.date_cycles(sp, 0.20)
print(spc.assign(months=(spc.duration / 21).round(1), start=spc.start.dt.date, end=spc.end.dt.date)
      [["phase", "start", "end", "months", "amplitude", "censored"]].to_string(index=False))
"""),
        md("""
## 3 · Why the null is the whole test

A ±20% filter needs a 20% rise to confirm a bull and a 20% fall from the high to end it, so a
young bull rarely ends: hazard starts low and rises. That is positive duration dependence
**manufactured by measurement**. Below, 300 memoryless random walks with the FF tape's monthly drift
and vol, dated by the same rule, each fitted with the same censored Weibull.
"""),
        code("""
lr = np.log1p(ff["mkt"])
paths = st.simulate_log_returns("rw", len(lr), 300, {"mu": lr.mean(), "sigma": lr.std(ddof=1)}, seed=11)
ks, naive = [], []
for p in paths:
    d_, c_ = st._bulls_from_tps(st.turning_points(np.exp(np.cumsum(p)), 0.20), len(p))
    f_ = st.fit_weibull(d_, c_)
    if f_:
        ks.append(f_["k"]); naive.append(f_["k"] > 1 and f_["p_k1"] < 0.05)
ks = np.array(ks)
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(ks, bins=40, color="#8b949e")
ax.axvline(1, color="k", ls="--", label="k = 1 (memoryless)")
ax.set_xlabel("Weibull shape k of random-walk bulls"); ax.legend(); plt.show()
print(f"median k {np.median(ks):.2f}; share k>1 {np.mean(ks > 1):.0%}; naive LR test rejects k=1 (k>1) on {np.mean(naive):.0%} of memoryless paths")
"""),
        md("""
> \U0001F4A1 **In plain words.** Measure a coin-flip market with this ruler and it "dies of old age"
> most of the time. Testing real bulls against k = 1 would be testing the ruler.

## 4 · The teardown

### 4.1 · Censored Weibull on the real bulls
"""),
        code("""
d, c = st.bull_durations(lv, 0.20)
fit = st.fit_weibull(d, c)
boot = st.weibull_bootstrap(d, c, n_boot=1000, seed=1017)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in fit.items()})
print(f"cycle-bootstrap 95% CI for k: {boot['k_lo']:.2f} – {boot['k_hi']:.2f}; share of draws with k>1: {boot['share_k_gt_1']:.0%}")
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.hist(boot["ks"], bins=40, color="#1f6feb", alpha=0.8)
ax.axvline(fit["k"], color="k", lw=2, label=f"real k = {fit['k']:.2f}")
ax.axvline(1, color="k", ls="--")
ax.set_xlabel("bootstrap k (resampling whole bull markets)"); ax.legend(); plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** With eleven completed bulls the interval runs from "fragile when
> young" to "ages briskly". The data alone can't say much — which is exactly why the comparison has
> to be like-for-like.

### 4.2 · The life table
"""),
        code("""
res = st.duration_test(ff["mkt"], 0.20, n_sims=200, n_boot=200, rf=ff["rf"], seed=1017)
lt = res["life"].copy()
lt["random walk"] = res["nulls"]["rw"]["life"]["annual_hazard"]
lt["GARCH"] = res["nulls"]["garch"]["life"]["annual_hazard"]
print(lt[["at_risk", "ended", "censored", "annual_hazard", "random walk", "GARCH"]].to_string())
x = np.arange(len(lt))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(x, lt.annual_hazard * 100, "o-", lw=2.5, color="#1f6feb", label="real")
ax.plot(x, lt["random walk"] * 100, "s--", color="#8b949e", label="random-walk null (pooled)")
ax.plot(x, lt["GARCH"] * 100, "^--", color="#dab617", label="GARCH(1,1)-t null (pooled)")
ax.set_xticks(x); ax.set_xticklabels(lt.index); ax.set_ylabel("P(bull ends within a year | age), %")
ax.set_xlabel("bull age"); ax.legend(); plt.show()
"""),
        md("""
The real hazard rises over the first three years and then flattens — below the random-walk curve
throughout (real bulls are *longer* than random-walk bulls) and close to the GARCH curve, whose
hazard *falls* with age because volatility clustering produces long calm bulls.

### 4.3 · The real k against both nulls
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, kind, col in ((axes[0], "rw", "#8b949e"), (axes[1], "garch", "#dab617")):
    n = res["nulls"][kind]
    ax.hist(n["table"]["k"].dropna(), bins=35, color=col)
    ax.axvline(fit["k"], color="#1f6feb", lw=2.5, label=f"real k {fit['k']:.2f}")
    ax.set_title(f"{kind.upper()} null: median k {n['k_null_median']:.2f}, p = {n['p_k']:.2f}")
    ax.legend()
plt.tight_layout(); plt.show()
print("GARCH params (tape drift imposed):", {k: round(v, 5) for k, v in res["nulls"]["garch"]["params"].items()})
"""),
        md(f"""
Pinned run (1,000 paths each): p = **{R['p_k_rw']:.2f}** vs the random walk, **{R['p_k_garch']:.2f}** vs
GARCH. H₁ fails against both.

> \U0001F4A1 **In plain words.** Real bulls sit right in the middle of what memoryless markets
> produce. Nothing to explain.

### 4.4 · Sensitivity to the ruler
"""),
        code("""
gp = res["nulls"]["garch"]["params"]
rows = []
for label, up, down in (("±15%", .15, .15), ("±25%", .25, .25), ("+20%/−15%", .20, .15)):
    r_ = st.duration_test(ff["mkt"], up, down, n_sims=150, n_boot=200, with_prediction=False,
                          garch_params=gp, seed=1017)
    rows.append({"rule": label, "bulls": r_["n_bulls"], "median_m": r_["median_bull"], "k": r_["fit"]["k"],
                 "k_rw": r_["nulls"]["rw"]["k_null_median"], "p_rw": r_["nulls"]["rw"]["p_k"],
                 "k_garch": r_["nulls"]["garch"]["k_null_median"], "p_garch": r_["nulls"]["garch"]["p_k"]})
print(pd.DataFrame(rows).to_string(index=False))
"""),
        md(f"""
Across all four rules in the pinned run, the smallest p against either null is
{R['sens_min_p']:.2f}. The ±15% and ±25% rules give k *below* 1 on the real tape.

### 4.5 · Real-time age — no look-ahead

`realtime_state` runs the filter month by month; the age at *t* is months since the trough that
started the **confirmed** bull. Truncating the tape leaves every earlier state unchanged (unit-tested).
"""),
        code("""
rt = st.realtime_state(lv, 0.20)
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(rt.index, rt.age / 12, color="#1f6feb", lw=1)
ax.set_ylabel("real-time bull age, years"); plt.show()
"""),
        md("""
### 4.6 · Does age predict the next 12 months?
"""),
        code("""
pr = res["pred_raw"]
dd = st.predictive_regression(ff["mkt"], ff["rf"], target="drawdown")
print(pd.DataFrame([
    {"target": "next-12m excess log return", "slope/yr": pr["slope"], "NW t (18 lags)": pr["t"],
     "non-overlap t": pr["t_nonoverlap"], "n": pr["n"], "n non-overlap": pr["n_nonoverlap"]},
    {"target": "next-12m max drawdown", "slope/yr": dd["slope"], "NW t (18 lags)": dd["t"],
     "non-overlap t": dd["t_nonoverlap"], "n": dd["n"], "n non-overlap": dd["n_nonoverlap"]}]).to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 3.8))
tn = res["nulls"]["rw"]["table"]["pred_t"].dropna()
ax.hist(tn, bins=35, color="#8b949e", label="random-walk paths, same regression")
ax.axvline(pr["t"], color="#1f6feb", lw=2.5, label=f"real t {pr['t']:+.2f}")
ax.axvline(-2, color="k", ls="--", lw=1)
ax.set_xlabel("HAC t of the age slope"); ax.legend(); plt.show()
print(f"simulated p (RW) {res['nulls']['rw']['p_pred']:.2f}; share of null paths with t <= -2: {(tn <= -2).mean():.0%}")
"""),
        md(f"""
> \U0001F4A1 **In plain words.** Age is a slowly-ticking clock, and regressions on slow clocks
> produce big-looking t-stats by accident — close to one random walk in ten gives t ≤ −2 here. The real
> t of {R['pred_t']:+.2f} is unremarkable even before that correction.

Next-month excess return by real-time state — what a de-risker gives up:
"""),
        code("""
W = st.age_rule_weights(lv, 0.20)
nxt = ff["mkt_rf"].shift(-1)
inb = W.age.notna() & W.median_age.notna()
old, young = inb & (W.age > W.median_age), inb & (W.age <= W.median_age)
rest = ~inb & W.median_age.notna()
rows = []
for name, m in (("young bull", young), ("old bull", old), ("bear / unconfirmed", rest)):
    x = nxt[m].dropna()
    rows.append({"state": name, "months": len(x), "mean (ann.)": x.mean() * 12,
                 "vol (ann.)": x.std() * np.sqrt(12), "Sharpe": x.mean() / x.std() * np.sqrt(12)})
print(pd.DataFrame(rows).to_string(index=False))
"""),
        md("""
Old bulls paid *at least* as well as young ones; the months the rule would sit fully invested
(bears, unconfirmed recoveries) paid the least. The timing is backwards before costs.

## 5 · The verdict
"""),
        code("""
h = dict(n_years=len(ff) / 12, n_bulls=res["n_bulls"], k=fit["k"], k_lo=boot["k_lo"], k_hi=boot["k_hi"],
         k_null_rw=res["nulls"]["rw"]["k_null_median"], k_null_garch=res["nulls"]["garch"]["k_null_median"],
         p_k_rw=res["nulls"]["rw"]["p_k"], p_k_garch=res["nulls"]["garch"]["p_k"],
         p_pred_rw=res["nulls"]["rw"]["p_pred"], p_pred_garch=res["nulls"]["garch"]["p_pred"],
         pred_t=pr["t"], pred_slope=pr["slope"], cost_bps=10.0)
start = W.median_age.first_valid_index()
bt = st.age_rule_backtest(ff["mkt"], ff["rf"], W.weight, 10.0)
bt = bt[bt.index > start]
s = st.summarize_backtest(bt)
b = st.sharpe_diff_bootstrap(bt.net_x, bt.bh_x, n_boot=1000)
h.update(sharpe_bh=s["sharpe_bh"], sharpe_net=s["sharpe_net"], sharpe_diff=b["diff"], sharpe_lo=b["lo"],
         sharpe_hi=b["hi"], p_sharpe=b["p"], tw_bh=s["tw_bh"], tw_net=s["tw_net"],
         share_derisked=s["share_derisked"], sp_sharpe_diff=-0.087)   # S&P leg: see §6 / results.md
v = st.verdict(h)
print("Signal:", v["signal"], "| Tradability:", v["trad"])
"""),
        md(f"""
Live re-run of `strategy.verdict` on this notebook's smaller simulation — it reproduces the pinned
stamps: **Signal {R['signal']} · Tradability {R['trad']}**. The pinned rationale is in
[`docs/results.md`](../docs/results.md#verdict).

## 6 · Could you trade it?
"""),
        code("""
rows = []
for cbps in (0.0, 10.0, 25.0):
    bt_ = st.age_rule_backtest(ff["mkt"], ff["rf"], W.weight, cbps)
    bt_ = bt_[bt_.index > start]
    s_ = st.summarize_backtest(bt_)
    b_ = st.sharpe_diff_bootstrap(bt_.net_x, bt_.bh_x, n_boot=500)
    rows.append({"cost bp (one-way)": cbps, "Sharpe B&H": s_["sharpe_bh"], "Sharpe rule gross": s_["sharpe_gross"],
                 "Sharpe rule net": s_["sharpe_net"], "Δ net": b_["diff"], "CI lo": b_["lo"], "CI hi": b_["hi"],
                 "$1 B&H": s_["tw_bh"], "$1 rule net": s_["tw_net"]})
print(pd.DataFrame(rows).to_string(index=False))
cm = st.constant_mix(bt.bh, bt.rf, s["avg_weight"])
print(f"\\nexposure-matched constant mix ({s['avg_weight']:.0%} equity, costless): Sharpe {st.sharpe(cm - bt.rf):.3f}, $1 -> ${(1 + cm).prod():,.0f}")
print(f"de-risked {s['share_derisked']:.0%} of months; one-way turnover {s['turnover_yr']:.0%} NAV/yr; "
      f"vol {s['vol_bh']:.1%} -> {s['vol_net']:.1%}; max DD {s['mdd_bh']:.0%} -> {s['mdd_net']:.0%}")
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy((1 + bt.bh).cumprod(), color="k", label="buy & hold")
ax.semilogy((1 + bt.net).cumprod(), color="#c0392b", label="de-risk old bulls (net)")
ax.semilogy((1 + cm).cumprod(), color="#8b949e", ls="--", label="constant mix, same avg exposure")
ax.legend(); plt.show()
"""),
        md(f"""
> \U0001F4A1 **In plain words.** The rule lowers risk the way any cash holding does, but it times
> it badly — it is worse than simply holding the same average amount of cash all the time
> (${R['tw_cm']:,} vs ${R['tw_net']:,} from $1). Costs barely matter: turnover is tiny. The loss
> is the timing.

Out-of-sample: daily S&P 500 **price** index 1990 → 2018-11, with the age threshold seeded by the
median of FF bulls confirmed before 1990 (cash = daily-ised T-bill). Pinned run: Sharpe gap
{R['sp_sharpe_diff']:+.3f}, rule worse. Capacity is not the issue — this is one index, monthly; the
rule fails on returns, not on size.

### Machinery proof (synthetic — not market evidence)

The same pipeline on `data.synthetic_monthly`: planted ageing (`signal_strength=1`) vs the exact
random-walk null (`0`).
"""),
        code("""
rows = []
for s_ in (1.0, 0.0):
    dfs, tr = data.synthetic_monthly(n_months=len(ff), signal_strength=s_, seed=1017)
    rs = st.duration_test(dfs["mkt"], n_sims=100, n_boot=100, kinds=("rw",), rf=dfs["rf"])
    rows.append({"signal_strength": s_, "bulls": rs["n_bulls"], "k": rs["fit"]["k"],
                 "null k": rs["nulls"]["rw"]["k_null_median"], "p(k)": rs["nulls"]["rw"]["p_k"],
                 "pred t": rs["pred_raw"]["t"], "p(pred)": rs["nulls"]["rw"]["p_pred"]})
print(pd.DataFrame(rows).to_string(index=False))
"""),
        md("""
The harness flags planted ageing on both tests and stays quiet on the matched null — so the null
result on the real tape is about markets, not a blind detector.

## 7 · Going further

- **Power is the binding constraint.** Eleven completed bulls in 92 years. A fair summary is "no
  evidence of excess ageing", not "memorylessness proven"; a modest true effect would be invisible.
  International indices (pooled hazards across markets) are the obvious way to add cycles.
- **Valuation, not age.** The plausible channel for late-cycle fragility is time-varying expected
  returns. Replacing *age* with a valuation state in §4.6 is a different study.
- **Bry–Boschan / Pagan–Sossounov dating** with minimum phase and cycle lengths would remove the
  shortest phases on real *and* null tapes; it is the natural fork of `date_cycles`.
- **Duration-dependent Markov switching** (Maheu & McCurdy 2000) models the hazard inside a
  return-generating process instead of on dated cycles — a stronger but more model-dependent test.
- Challenge the nulls: add a slowly mean-reverting expected return to the random walk and see
  whether that alone produces real-looking ageing.
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
