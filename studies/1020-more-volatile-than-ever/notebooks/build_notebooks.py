"""Notebook builder for Study 1020 — More Volatile Than Ever?

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    cd notebooks
    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks walk the same seven desk beats (METHODOLOGY.md) at two altitudes: 01 is the
plain-language layer, 02 carries the inference. The headline numbers quoted in prose live in
ONE place — the ``REAL`` dict and the verdict strings below — and mirror ``docs/results.md``
(regenerate both whenever ``examples/verify.py`` is re-run). The code cells recompute every
figure from the frozen tapes; synthetic cells are labelled as synthetic.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# The real run, pasted from docs/results.md (verify.py, as-of 2021-12-31).
# --------------------------------------------------------------------------- #
REAL = {
    "as_of": "2021-12-31",
    "signal": "None", "trad": "Fragile", "myth": "Busted",
    "fp_monthly": "5bf075a78851", "fp_daily": "6801b52cecb2", "fp_ohlc": "317bd3262050",
    "century_decade": "-5.6%", "century_t_nw": "-2.13", "century_p_kvb": "0.26",
    "daily_decade": "+6.5%", "daily_t_nw": "+0.70", "daily_p_kvb": "0.61",
    "kvb_crit": "5.91",
    "thirties_vol": "35.3%", "remembered_vol": "15.8%", "thirties_ratio": "2.2",
    "thirties_p": "0.0003", "thirties_down10": "15", "remembered_down10": "2",
    "points_decade": "+95%", "pct_decade": "+4%",
    "top_points_last_decade": "95%", "top_pct_last_decade": "25%",
    "extremes_total": "129", "extremes_p_perm": "0.58",
    "q_ann_lr": "0.466", "q_ann_recent": "0.372", "q_ann_trend": "0.329",
    "q_ann_window": "0.345", "q_ann_blend": "0.318",
    "q_mon_lr": "0.684", "q_mon_recent": "0.532", "q_mon_trend": "0.839",
    "q_mon_window": "0.762", "q_mon_blend": "0.506",
    "syn_fire_planted": "80%", "syn_fire_null": "2%",
}

# Verdict prose, verbatim from docs/results.md (strategy.verdict on the real run).
SIGNAL_WHY = 'Over 91 complete years (1927-2017, Fama-French total return), the trend in log realised volatility is **-5.6% per decade** — the point slope is actually *negative* (Newey-West *t* = -2.13; KVB fixed-b *t* = -2.85, p = 0.26; block-bootstrap p = 0.02), and what slope there is comes from the 1930s sitting at the start. On the daily S&P 500 price index (1990-2021) it is +6.5% per decade (NW *t* = +0.70, KVB p = 0.61); the independent intraday-range estimator (1999-2018) gives -26.2% (NW *t* = -3.06) over a window that opens in the dot-com bust. Extreme days against a fixed long-run σ: 129 sessions beyond 3σ, a fitted trend of +40% per decade that does not survive a test respecting crisis clustering (HAC z = +1.49, block-permutation p = 0.58) — the count is two bursts, 2008-09 and 2020, not a slope. None of the start years from 1927 to 1998 gives a significant rise to 2017. Volatility arrives in storms and then calms down; it does not climb.'
TRAD_WHY = "Framed as a risk manager's choice of next-period volatility, scored out of sample by QLIKE. Century tape (annual, 71 years): LONG-RUN 0.466, RECENT 0.372, TREND 0.329, and a plain 20-year mean with no slope 0.345. Daily tape (monthly, 324 months): LONG-RUN 0.684, RECENT 0.532, TREND 0.839, WINDOW 0.762. The trend-extrapolation rule — the claim turned into a forecast — beats the long-run average on the century (DM *t* = -1.51, not significant) and loses to it monthly (DM *t* = +0.81); against the same window *without* the slope it scores DM *t* = -0.24 (annual) and +0.54 (monthly). Where TREND wins it is because a shorter memory forgets the 1930s, not because volatility rises. The robust lesson is the opposite of the claim's: the best forecasts here mix recent and long-run (BLEND vs LONG-RUN DM *t* = -4.39 annual, -2.30 monthly) — volatility clusters and then mean-reverts. Nothing is traded, so no costs are charged; the stamp grades the trend rule by the pre-registered bar — Fragile rather than Mirage only because that bar asks TREND to beat the long-run average on one tape on a point estimate, which it does by forgetting, not by extrapolating."
ONE_SENTENCE = 'A century of US market volatility shows no upward trend (-5.6% per decade, KVB p = 0.26), the 1930s were 2.2× as volatile as 2008-2017, and the record crashes in the headlines are point moves that grow with the index (+95% per decade in points vs +4% in percent) — a risk manager should mix recent and long-run volatility, not extrapolate a rise.'


BADGE = {"Real": "2ea44f", "Weak": "dab617", "Mixed": "dab617", "None": "c0392b",
         "Investable": "2ea44f", "Fragile": "dab617", "Mirage": "c0392b"}


def T(text: str) -> str:
    """Fill ``<<key>>`` tokens from REAL."""
    for k, v in REAL.items():
        text = text.replace(f"<<{k}>>", str(v))
    return text


def badges() -> str:
    s, t, m = REAL["signal"], REAL["trad"], REAL["myth"]
    return (f"![Signal: {s}](https://img.shields.io/badge/Signal-{s}-{BADGE[s]}?style=flat-square)\n"
            f"![Tradability: {t}](https://img.shields.io/badge/Tradability-{t}-{BADGE[t]}"
            f"?style=flat-square)\n"
            f"![More volatile than ever?: {m}](https://img.shields.io/badge/"
            f"More_volatile_than_ever%3F-{m.replace(' ', '_')}-8b949e?style=flat-square)")


def md(text):
    return new_markdown_cell(T(text.strip("\n").rstrip()))


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


BOOT = r"""
import sys, os, io, contextlib
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
from volhistory import data, strategy as st
print("tapes available:", data.have_real(), "| study as-of", data.AS_OF)
"""

LOAD = r"""
m = data.load_monthly()      # Fama-French monthly, TOTAL return, 1926-07 -> 2018-11
d = data.load_daily()        # S&P 500 daily, PRICE index, 1990 -> 2021 (partial 2022 dropped)
o = data.load_ohlc()         # S&P 500 daily OHLC, PRICE index, 1999 -> 2018
for p in data.provenance():
    print(f"{p['tape']:8s} {p['label']:55s} {p['first']} -> {p['last']}  fp {p['fingerprint']}")
rv_c = st.annual_rv_from_monthly(m, data.FIRST_FULL_YEAR_MONTHLY, data.LAST_FULL_YEAR_MONTHLY)
rv_d = st.annual_rv_from_daily(d)
rv_p = st.annual_rv_parkinson(o)
"""

ERA_COLORS = '["#e8eef7", "#f4f4f4", "#e8eef7", "#f4f4f4"]'


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md("""
# More Volatile Than Ever? \U0001F3A2
### A century of the US stock market against the most repeated sentence in financial news

""" + badges() + """

"Markets have never been this volatile." It has been said about program trading, day traders,
high-frequency algorithms, ETFs and Reddit. We put **ninety-one complete years** of the US market
on the bench, plus thirty years of daily S&P 500 data, and asked whether volatility is actually
rising, whether crash days are getting more common — and if not, why it feels that way.

> \U0001F4D3 **This is the plain-language layer.** The trend tests, the fixed-b inference and the
> forecast race in full are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> ⚠️ **Not investment advice.** Every chart is computed from frozen, fingerprinted tapes
> (as-of <<as_of>>); the one synthetic section is labelled as such.
"""),
        code(BOOT),
        md("""
## 0 · The answer first

| Question | Answer (real tape) |
|---|---|
| Is market volatility trending up over a century? | **No.** <<century_decade>> per decade — slightly *down*, and not reliably either way. |
| Was the decade you remember unusually wild? | The 1930s were **<<thirties_ratio>>×** as volatile as 2008-2017. |
| Are crash days getting more frequent? | They come in bursts (2008-09, 2020), not on a slope. |
| Then why does it feel that way? | Headlines quote **points**. Point moves grow **<<points_decade>> per decade**; percent moves **<<pct_decade>>**. |
| What should a risk manager do? | Mix recent and long-run volatility — don't extrapolate a rise. |

**Signal: <<signal>> · Tradability: <<trad>> · More volatile than ever? <<myth>>.**
"""),
        md("""
## 1 · The claim

The strong version, stated fairly: markets today are wired differently. Algorithms trade in
microseconds and pull liquidity at the first sign of stress; ETFs let a whole market be sold in
one click; social media turns a rumour into a stampede in minutes. Every one of those changes
plausibly *amplifies* moves. So volatility should be higher than in the slow, human markets of
the past — and the extreme days, the 1987s and 2020s, should be getting more common.

It is a serious argument. The 2010 flash crash really did happen in minutes, and studies of
ETF ownership do find it raises the volatility of individual stocks.
"""),
        md("""
## 2 · So what?

If it were true, every risk number calibrated on history would be too low, and should be scaled
up by an amount that grows each year. Pension funds, banks and anyone sizing a position on
"normal" volatility would be systematically under-protected. If it is false, the people
extrapolating it are paying for insurance they do not need — and, worse, are forecasting the
wrong thing.
"""),
        md("""
## 3 · How we'd know

Decided before looking:

- **A trend** in volatility means a slope in log volatility year after year that survives tests
  which know volatility is *sticky* (a calm year is usually followed by a calm year — that
  stickiness fools naive trend lines).
- **More extreme days** means the count of 3σ days per year trends up, by a test that knows
  crashes arrive in clusters.
- **"Mirage"** would be: no tape shows a significant rise, the trend-extrapolating forecast does
  no better than the long-run average, and the feeling is explained by something else.

> \U0001F52C **For the quants.** The bar is a positive slope with Newey-West *t* ≥ 2 *and* a
> Kiefer-Vogelsang-Bunzel fixed-b p < 0.05 (critical value <<kvb_crit>>, simulated), plus a
> residual block bootstrap. The rule is `strategy.verdict`.
"""),
        md("""
## 4 · The teardown

### 4.1 · Ninety-one years of volatility
"""),
        code(LOAD),
        code(r"""
fig, ax = plt.subplots(figsize=(11, 5))
for (name, y0, y1), c in zip(data.ERAS, """ + ERA_COLORS + r"""):
    ax.axvspan(y0, y1 + 0.99, color=c, zorder=0)
    ax.text((y0 + y1) / 2, 0.62, name, ha="center", fontsize=9, color="#555")
ax.plot(rv_c.index, rv_c * 100, color="#1f6feb", lw=1.8, label="US market, monthly data (total return)")
ax.plot(rv_d.index, rv_d * 100, color="#c0392b", lw=1.4, alpha=0.8, label="S&P 500, daily data (price index)")
tr = st.trend_test(np.log(rv_c), n_boot=499)
yrs = np.array(rv_c.index, dtype=float)
fit = np.exp(np.polyval(np.polyfit(yrs, np.log(rv_c), 1), yrs))
ax.plot(yrs, fit * 100, "k--", lw=1.5, label=f"century trend: {tr['pct_per_decade']:+.1%} per decade")
ax.set_ylabel("realised volatility, % a year"); ax.set_ylim(0, 65)
ax.legend(loc="upper right", fontsize=9)
ax.set_title("No staircase upward: storms, then calm")
plt.show()
print(f"century trend {tr['pct_per_decade']:+.1%}/decade, Newey-West t {tr['t_nw']:+.2f}, "
      f"fixed-b p {tr['p_kvb']:.2f}")
"""),
        md("""
The tallest spikes are in the **1930s**. The post-war decades were calm, the 1970s and 1987 woke
things up, and then 2000-02, 2008 and 2020 each spiked — and each calmed down. If you drew the
line from 1995 to 2008 you would see a terrifying rise; from 1932 to 2017, a decline. Neither is a
trend.
"""),
        md("""
### 4.2 · The decade you remember vs the 1930s
"""),
        code(r"""
dc = st.decade_contrast(m, (1930, 1939), (2008, 2017))
dv = st.decade_vols(m)
fig, ax = plt.subplots(figsize=(11, 4.5))
cols = ["#c0392b" if i == "1930s" else "#1f6feb" if i in ("2000s", "2010s") else "#8b949e" for i in dv.index]
ax.bar(dv.index, dv["vol"] * 100, color=cols)
ax.set_ylabel("annualised volatility, %")
ax.set_title("Volatility by decade (monthly total returns; the 1920s and 2010s are partial)")
plt.show()
print(f"1930-39: {dc['vol_a']:.1%}   2008-17: {dc['vol_b']:.1%}   ratio {dc['ratio']:.1f}x")
print(f"months the market fell 10% or more: {dc['n_down10_a']} vs {dc['n_down10_b']}")
"""),
        md("""
The ten years that contain the 2008 crisis — the most frightening stretch most readers have
lived through — were **less than half** as volatile as the 1930s, which had
**<<thirties_down10>> months** with a fall of 10% or more against <<remembered_down10>>. Memory
is a bad yardstick: we compare today with the last thing we remember, not with the record.
"""),
        md("""
### 4.3 · Why it *feels* more volatile: points versus percent
"""),
        code(r"""
pp = st.points_vs_percent(d)
y = pp["yearly"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(y.index, y["mean_abs_points"], color="#c0392b")
axes[0].set_title("average daily move in INDEX POINTS")
axes[1].bar(y.index, y["mean_abs_pct"] * 100, color="#1f6feb")
axes[1].set_title("average daily move in PERCENT")
axes[1].set_ylabel("%")
plt.tight_layout(); plt.show()
print(f"points: {pp['trend_points']['pct_per_decade']:+.0%} per decade;  "
      f"percent: {pp['trend_pct']['pct_per_decade']:+.0%} per decade")
print(f"of the 20 biggest POINT drops, {pp['top_points_share_last_decade']:.0%} are in "
      f"{pp['last_decade_start']}-2021; of the 20 biggest PERCENT drops, "
      f"{pp['top_pct_share_last_decade']:.0%}")
"""),
        md("""
This is the whole trick. A 2% day when the S&P is at 400 is 8 points; at 4,000 it is 80. The
*risk* is identical. The *headline* — "biggest point drop in history" — is ten times bigger. An
index that grows sets point records automatically, and the S&P 500 grew about tenfold over this
tape. Every "record crash" story you have read since 2018 is, first, a statement about how high
the market had climbed.

> \U0001F52C **For the quants.** ΔP = P × r, so the trend in log|ΔP| is the trend in log P plus
> the trend in log|r|. The first is enormous, the second is nil.
"""),
        md("""
### 4.4 · Are extreme days getting more frequent?
"""),
        code(r"""
lr = st.log_returns_from_price(d)
B = (("1990-99", 1990, 1999), ("2000-09", 2000, 2009), ("2010-19", 2010, 2019), ("2020-21", 2020, 2021))
tab = st.count_table(st.extreme_days(lr, 3.0, "fixed"), B)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(tab.index, tab["rate"], color="#8b949e",
       yerr=[tab["rate"] - tab["ci_lo"], tab["ci_hi"] - tab["rate"]], capsize=6)
ax.set_ylabel("days beyond 3σ per 1,000 sessions")
ax.set_title("S&P 500 days moving more than 3 long-run σ (95% exact intervals)")
plt.show()
print(tab.round(1).to_string())
"""),
        md("""
The 2000s are high because of 2008-09; 2020-21 is high because of March 2020 (and is only two
years, hence the huge interval). The 2010s were *quieter* than the 2000s. That is two bursts, not
a slope — and once the test is told that crash days come in clusters, the apparent trend is
indistinguishable from chance (p = <<extremes_p_perm>>).
"""),
        md("""
## 5 · The verdict

**Signal: <<signal>>.** No tape shows a significant rise in volatility: the century trend is
<<century_decade>> per decade, the daily 1990-2021 tape <<daily_decade>> with *t* = <<daily_t_nw>>.

**Tradability: <<trad>>.** A rule that extrapolates the rise is a poor risk forecast; where it
seems to help, the help comes from forgetting the 1930s, not from the trend.

**More volatile than ever? <<myth>>.** The 1930s were <<thirties_ratio>>× as volatile as
2008-2017.

> **In one sentence:** """ + "<<one_sentence>>" + """
"""),
        md("""
## 6 · Could you use it?

The useful version of the question is a risk manager's: *what volatility should I plan for next
year?* We ran a race, out of sample, between the **long-run average**, **recent history**, and
**the trend extrapolated** — the claim turned into a rule.
"""),
        code(r"""
fa = st.forecast_race(rv_c ** 2, recent=1, trend_window=20, burn=20)
fm = st.forecast_race(st.monthly_rv_from_daily(d), recent=12, trend_window=60, burn=60)
order = ["LONG-RUN", "RECENT", "TREND", "BLEND"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, f, title in ((axes[0], fa, "next YEAR, 1947-2017"), (axes[1], fm, "next MONTH, 1995-2021")):
    q = [f["mean_qlike"][k] for k in order]
    ax.bar(order, q, color=["#8b949e", "#1f6feb", "#c0392b", "#2ea44f"])
    ax.set_title(f"forecast error (QLIKE, lower is better): {title}", fontsize=10)
plt.tight_layout(); plt.show()
"""),
        md("""
Recent history helps, because volatility is sticky in the short run. The long-run average helps,
because storms end. The best forecast in both races **mixes** them (green). The trend rule is the
worst forecast month-to-month, and its apparent edge year-to-year disappears against a plain
20-year average with no slope at all. There is nothing to trade here; there is a calibration
mistake to avoid.
"""),
        md("""
## 7 · Going further \U0001F6AA

- **Individual stocks are a different story.** Campbell, Lettau, Malkiel & Xu (2001) found
  *idiosyncratic* volatility rising through the 1990s while market volatility did not. A fork
  with a stock panel could ask whether that trend survived.
- **Intraday is a different story too.** Flash crashes are minutes long; daily and monthly data
  cannot see them. A fork with intraday data could test whether *within-day* extremes rose
  after 2010.
- **Other countries.** The US had the calmest century of any major market; the claim deserves a
  test on markets that did not.
- **Challenge us:** change the eras, the start year or the forecast window in
  `examples/verify.py` and see whether anything flips.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md("""
# More Volatile Than Ever? — a quantitative teardown \U0001F52C
### Trend tests under persistence · extreme-day counts · points vs percent · a forecast race

""" + badges() + """

The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Annual log
realised volatility is tested for a trend three ways — Newey-West, Kiefer-Vogelsang-Bunzel
fixed-b, residual block bootstrap — on a century of monthly total returns and two daily S&P 500
tapes; extreme days are counted against fixed and trailing σ with exact Poisson intervals and a
block-permutation trend test; and the claim, turned into a forecasting rule, races the
long-run and recent alternatives out of sample.

> ⚠️ **Not investment advice.** Real numbers come from frozen bundled tapes (fingerprints
> `<<fp_monthly>>` / `<<fp_daily>>` / `<<fp_ohlc>>`, as-of <<as_of>>; see
> [`docs/results.md`](../docs/results.md)). §4.7 is a **synthetic** control and says so.
>
> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT),
        code(LOAD),
        md("""
## 0 · Verdict (real tape), up front

| Axis | Stamp | Why |
|---|---|---|
| **Signal** | <<signal>> | <<signal_why>> |
| **Tradability** | <<trad>> | <<trad_why>> |
| **More volatile than ever?** | <<myth>> | The 1930s were <<thirties_ratio>>× as volatile as 2008-2017 (p = <<thirties_p>>). |

> \U0001F4A1 **In plain words.** Volatility comes in storms that end. Nothing in a century of
> data says the storms are getting stronger; the headlines measure them in points.
"""),
        md("""
## 1 · The claim, and the hypotheses that would make it true

- **H₁ (trend).** Log realised volatility of the US market has a positive linear trend,
  1927-2017, robust to persistence.
- **H₂ (recent tape).** The same holds on daily S&P 500 data, 1990-2021, and on an independent
  range-based estimator, 1999-2018.
- **H₃ (tails).** The yearly rate of |r| > 3σ days trends up (fixed σ), robust to crisis
  clustering.
- **H₄ (perception).** Point moves trend up even under constant percent volatility — the
  mechanism that would make H₁-H₃ *feel* true when false.
- **H₅ (forecast).** Extrapolating the trend beats the long-run average as a risk forecast.
"""),
        md("""
## 2 · So what?

Under H₁ every historical risk number is biased low by a growing amount and VaR models should
be trend-adjusted. Under its negation the right prior is a stationary, persistent volatility —
which is what every GARCH and HAR model already assumes — and trend-adjusting adds error.
"""),
        md("""
## 3 · How we'd know — the tests and the pre-registered bar

A persistent stationary series wanders, and an OLS trend through it over-rejects badly. The
Newey-West correction with a rule-of-thumb bandwidth helps but still over-rejects at volatility's
persistence. The fixed-b approach (Kiefer, Vogelsang & Bunzel 2000; Kiefer & Vogelsang 2002)
sets the Bartlett bandwidth to the whole sample; the statistic's null distribution is then
non-normal but pivotal, and its critical values are simulated below rather than borrowed.
"""),
        code(r"""
null = st.kvb_null_distribution()
print(f"KVB fixed-b |t| for a trend slope: 90% {np.quantile(null, .90):.2f}, "
      f"95% {np.quantile(null, .95):.2f}, 99% {np.quantile(null, .99):.2f}  (normal: 1.64 / 1.96 / 2.58)")
print()
print(st.verdict.__doc__)
"""),
        md("""
> \U0001F4A1 **In plain words.** Because volatility is sticky, a *t* of 3 on a volatility trend
> is much weaker evidence than a *t* of 3 on, say, an average return. The fixed-b test charges
> for that stickiness honestly; its 5% hurdle is about <<kvb_crit>>, not 1.96.
"""),
        md("""
## 4 · The teardown

### 4.1 · Trend tests on three tapes and two estimators
"""),
        code(r"""
rows = {}
rows["century, monthly TR (1927-2017)"] = st.trend_test(np.log(rv_c))
rows["S&P daily close-to-close (1990-2021)"] = st.trend_test(np.log(rv_d))
rows["S&P daily close-to-close, arch (1999-2018)"] = st.trend_test(np.log(st.annual_rv_from_daily(o["close"])))
rows["S&P Parkinson range (1999-2018)"] = st.trend_test(np.log(rv_p))
tt = pd.DataFrame(rows).T[["n", "pct_per_decade", "t_ols", "t_nw", "nw_lags", "t_kvb", "p_kvb", "p_boot", "rho1"]]
print(tt.astype(float).round(3).to_string())
print("\npasses the pre-registered UP bar:", {k: st.passes(v) for k, v in rows.items()})
"""),
        md("""
The century slope is negative: NW *t* = <<century_t_nw>>, bootstrap p ≈ 0.02 — but the fixed-b
p is <<century_p_kvb>>. The decline is the 1930s at the start of the window; the honest test
will not certify it either way. The daily tape slopes up at <<daily_decade>> per decade with
NW *t* = <<daily_t_nw>> — nothing. The 1999-2018 window slopes *down* on both estimators, a
start-date effect (it opens in the dot-com bust).

> \U0001F4A1 **In plain words.** Pick the window and you pick the answer. That is the signature
> of a series without a trend.
"""),
        md("""
### 4.2 · Start-year sensitivity
"""),
        code(r"""
starts = st.trend_from_every_start(np.log(rv_c), min_len=20)
fig, ax = plt.subplots(figsize=(11, 4.5))
c = np.where(starts["slope"] > 0, "#c0392b", "#1f6feb")
ax.bar(starts.index, starts["pct_per_decade"] * 100, color=c)
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("first year of the window (all windows end in 2017)")
ax.set_ylabel("fitted trend, % per decade")
plt.show()
up = (starts["slope"] > 0) & (starts["t_nw"] >= 2) & (starts["p_kvb"] < 0.05)
print(f"slope > 0 from {np.mean(starts['slope'] > 0):.0%} of start years; "
      f"significantly up (full bar) from {up.mean():.0%}")
"""),
        md("""
### 4.3 · Four eras
"""),
        code(r"""
era = st.era_table(m, data.ERAS)
print(era.round(4).to_string())
fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(range(len(era)), era["vol"] * 100,
            yerr=[(era["vol"] - era["ci_lo"]) * 100, (era["ci_hi"] - era["vol"]) * 100],
            fmt="o", capsize=6, color="#1f6feb", ms=8)
ax.set_xticks(range(len(era))); ax.set_xticklabels(era.index)
ax.set_ylabel("annualised vol, % (95% block-bootstrap CI)")
plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** The modern era (1990-2018) sits between the calm post-war decades
> and the turbulent 1970s-80s, and far below the Depression era. No within-era trend is
> significant either.
"""),
        md("""
### 4.4 · Extreme days: fixed σ vs trailing σ
"""),
        code(r"""
lr = st.log_returns_from_price(d)
B = (("1990-99", 1990, 1999), ("2000-09", 2000, 2009), ("2010-19", 2010, 2019), ("2020-21", 2020, 2021))
out = []
for k in (3, 4):
    for mode in ("fixed", "trailing"):
        f = st.extreme_days(lr, k, mode)
        tab = st.count_table(f, B)
        ct = st.count_trend(f, n_boot=999)
        out.append({"rule": f"|r| > {k}σ, {mode}", "total": int(ct["total"]),
                    **{b: round(tab.loc[b, "rate"], 1) for b in tab.index},
                    "trend/decade": ct["pct_per_decade"], "HAC z": ct["z_hac"], "block-perm p": ct["p_perm"]})
print("rates per 1,000 sessions")
print(pd.DataFrame(out).set_index("rule").round(3).to_string())
f3 = st.extreme_days(lr, 3.0, "fixed")
yearly = f3.groupby(f3.index.year).sum()
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.bar(yearly.index, yearly.values, color="#8b949e")
ax.set_title("days beyond 3 long-run σ, per year — two bursts, not a slope")
plt.show()
print("months beyond 3σ on the century tape, by decade:", st.monthly_extremes_by_decade(m)["count"].to_dict())
"""),
        md("""
Against a fixed σ the 2000s and 2020-21 dominate; against a trailing σ the rate of *surprising*
days drifts up across decades — calm regimes broken suddenly, which is a statement about the
volatility of volatility, not its level. The HAC *z* on the trailing counts can exceed 2; the
block permutation, which keeps each crisis cluster together, puts every version at p > 0.5. On
the century tape, 11 of the 16 three-sigma *months* are in the 1930s.

> \U0001F4A1 **In plain words.** Counting crash days and fitting a line treats 2008's dozens of
> wild days as dozens of independent pieces of evidence. They are one event.
"""),
        md("""
### 4.5 · Points versus percent: the decomposition
"""),
        code(r"""
pp = st.points_vs_percent(d)
tab = pd.DataFrame({k: pp[k] for k in ("trend_points", "trend_pct", "trend_level")}).T
print(tab[["pct_per_decade", "slope", "t_nw", "p_kvb"]].astype(float).round(4).to_string())
print(f"\nslope(points) = {pp['trend_points']['slope']:.4f}  vs  slope(level) + slope(percent) = "
      f"{pp['trend_level']['slope'] + pp['trend_pct']['slope']:.4f}")
print("record POINT drops:", [(str(i.date()), round(v)) for i, v in pp["records_points"].items()])
print("record PERCENT drops:", [(str(i.date()), round(v, 3)) for i, v in pp["records_pct"].items()])

# the mechanism, isolated: a constant-volatility index (SYNTHETIC)
g, _ = data.synthetic_returns(n_years=32, freq="D", signal_strength=0.0, logvol_sd=1e-9, drift=0.08, seed=7)
pg = st.points_vs_percent(g["price"], big_points=10.0)
print(f"\nSYNTHETIC constant-σ index: points {pg['trend_points']['pct_per_decade']:+.0%}/decade "
      f"(NW t {pg['trend_points']['t_nw']:+.1f}), percent {pg['trend_pct']['pct_per_decade']:+.1%}/decade")
"""),
        md("""
> \U0001F4A1 **In plain words.** Twelve "biggest point drop ever" days in 1995-2021, against five
> in percent — and four of the twelve came in three weeks of 2020. A market that grows ~7% a
> year makes its point moves grow ~7% a year with *no change in risk*. The synthetic index,
> volatility held exactly constant, reproduces the headline trend on its own.
"""),
        md("""
### 4.6 · Recency: the remembered decade against the 1930s
"""),
        code(r"""
dc = st.decade_contrast(m, (1930, 1939), (2008, 2017))
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in dc.items()})
"""),
        md("""
The decade with the global financial crisis was **<<remembered_vol>>** annualised against
**<<thirties_vol>>** for the 1930s (block-bootstrap p = <<thirties_p>> on the difference). The
availability heuristic (Tversky & Kahneman 1973) does the rest: the reference class is "what I
remember", not "what happened".
"""),
        md("""
### 4.7 · Synthetic control — can the bar see a trend that is there? (SYNTHETIC)

A machinery proof, never market evidence. Ninety-year monthly stochastic-volatility worlds with
persistent log volatility; ``signal_strength`` scales a planted trend that, at 1.0, triples
volatility across the sample.
"""),
        code(r"""
rows = [st.synthetic_detection(s, seeds=range(15)) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
pw = pd.DataFrame(rows).set_index("signal_strength")
print(pw[["fire_rate", "nw_rate", "ols_rate", "mean_slope", "true_slope"]].round(4).to_string())
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(pw.index, pw["fire_rate"] * 100, "o-", color="#2ea44f", lw=2, label="pre-registered bar (NW + fixed-b)")
ax.plot(pw.index, pw["ols_rate"] * 100, "s--", color="#c0392b", lw=1.5, label="naive OLS t ≥ 2")
ax.axhline(5, color="k", ls=":", lw=1)
ax.set_xlabel("signal_strength (1 = volatility triples over 90 years)"); ax.set_ylabel("% of worlds flagged")
ax.legend(fontsize=9); plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** The detector catches a planted rise in most worlds
> (<<syn_fire_planted>> at full strength in the 40-world run in `results.md`) and almost never
> invents one (<<syn_fire_null>> on the null). A real rise of the size the claim implies would
> have been seen.
"""),
        md("""
## 5 · The verdict, recomputed live from the tapes
"""),
        code(r"""
import importlib.util
spec = importlib.util.spec_from_file_location("verify", os.path.abspath("../examples/verify.py"))
verify = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)
with contextlib.redirect_stdout(io.StringIO()):
    h = verify.report()
v = h["_verdict"]
print(f"Signal: {v['signal']}   Tradability: {v['trad']}   More volatile than ever? {v['myth']}")
print("flags:", v["flags"])
print()
print(v["one_sentence"])
"""),
        md("""
**Signal: <<signal>>.** <<signal_why>>

**Tradability: <<trad>>.** <<trad_why>>
"""),
        md("""
## 6 · Could you trade it? The risk manager's race in detail

One period of lag: each forecast uses data to the end of period *t* and is scored on *t+1*.
QLIKE on variances; Diebold-Mariano with Newey-West (3 lags) on the loss differential.
"""),
        code(r"""
fa = st.forecast_race(rv_c ** 2, recent=1, trend_window=20, burn=20)
fm = st.forecast_race(st.monthly_rv_from_daily(d), recent=12, trend_window=60, burn=60)
for name, f in (("ANNUAL (century)", fa), ("MONTHLY (daily tape)", fm)):
    print(name, f["n"], "forecasts")
    print(pd.Series(f["mean_qlike"]).round(3).to_string())
    print(pd.DataFrame(f["dm"]).T[["mean_diff", "t", "p"]].astype(float).round(3).to_string())
    print()
F = fa["forecasts"]
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(F.index, np.sqrt(F["realised"]) * 100, "k-", lw=2, label="realised")
for k, c in (("LONG-RUN", "#8b949e"), ("RECENT", "#1f6feb"), ("TREND", "#c0392b"), ("BLEND", "#2ea44f")):
    ax.plot(F.index, np.sqrt(F[k]) * 100, color=c, lw=1.3, label=k)
ax.set_ylabel("annual vol, %"); ax.legend(fontsize=9, ncol=5)
ax.set_title("next-year volatility forecasts vs what happened (century tape)")
plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** The long-run average spends decades too high because it still
> remembers 1932; recent history over-reacts to every storm; the extrapolated trend does both.
> Half long-run, half recent wins both races (DM *t* ≈ −4.4 annual, −2.3 monthly against the
> long-run). Nothing is traded and no costs apply — the stamp grades the trend rule, which
> clears the pre-registered *Fragile* bar only on a non-significant point estimate that
> vanishes against a slope-free window.
"""),
        md("""
## 7 · Going further

- **Idiosyncratic volatility.** Campbell, Lettau, Malkiel & Xu (2001) — rising firm-level, flat
  market-level volatility through the 1990s. A stock-panel fork could re-test it after 2000.
- **Long memory.** Volatility's autocorrelations decay hyperbolically; a fractionally
  integrated null (rather than AR(1)) would make the trend tests even more conservative. A
  fork could run Vogelsang's (1998) t-PS or a sieve bootstrap.
- **Volatility of volatility.** The trailing-σ counts hint that calm regimes break more
  abruptly; a test on the dispersion of log RV, not its level, is the natural next study.
- **Intraday.** Flash-crash-style events are invisible at daily frequency; the HFT version of
  the claim needs tick data.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


def _meta():
    return {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


REAL.update({"signal_why": SIGNAL_WHY, "trad_why": TRAD_WHY, "one_sentence": ONE_SENTENCE})

if __name__ == "__main__":
    build_curious()
    build_quants()
