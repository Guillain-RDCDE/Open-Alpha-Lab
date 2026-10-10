"""Notebook builder for Study 1015 — The 1929 Overlay.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    python notebooks/build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks walk the same seven desk beats (METHODOLOGY.md) at two altitudes: 01 is the
plain-language layer, 02 carries the inference, the corrections and the nulls. The verdict
badges and every headline number quoted in prose live in ONE dict, ``REAL``, copied from
``docs/results.md`` (the fingerprinted run of ``examples/verify.py``) — never chosen by hand.
Cells that recompute on the real tapes say so; cells on synthetic tapes say so too.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Mirrors docs/results.md (as-of 2022-11-30; monthly fp c7f1e4140bfb, daily fp e223c581ae22).
REAL = {
    "signal": "None", "trad": "Mirage", "myth": "Confirmed",
    "n_combos": 126, "n_raw_sig": 5, "share_raw_sig": "4%", "holm_min_p": "1.00",
    "m_t": "+1.77", "m_corr": "+0.13", "m_r2": "−21.6%", "m_hit": "55.7%",
    "d_t": "−1.63", "d_corr": "−0.12", "d_r2": "−21.1%",
    "m_sharpe": "0.47", "m_bh": "0.52", "m_alpha_t": "+0.26",
    "d_sharpe": "0.22", "d_bh": "0.34", "d_alpha_t": "−1.69",
    "rc_m": "1.00", "rc_d": "1.00",
    "real_match": "0.946", "null_match": "0.903", "volpath_match": "0.919",
    "block_match": "0.931", "rw_share_09": "about half",
    "p_pair_07": "18%", "p_pair_09": "2.5%",
    "tmpl_share_09": "29%", "tmpl_null_09": "15%", "crash_match": "5.3%", "crash_all": "4.3%",
    "fwd_match": "+5.9%", "fwd_all": "+11.2%", "tmpl_trail_corr": "0.82",
    "tmpl_orth_t": "−1.42", "showcase_corr": "0.952", "showcase_fwd": "+12.0%",
    "worst": "daily price path, L=63, h=63, k=5", "worst_t": "−4.31",
    "worst_holm2": "0.002",
    "syn_power": "100%", "syn_size": "0%",
}

SIGNAL_BADGE = "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)"
TRAD_BADGE = ("![Tradability: Mirage]"
              "(https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)")
MYTH_BADGE = ("![Shape check: Confirmed]"
              "(https://img.shields.io/badge/Shape_check-Confirmed-8b949e?style=flat-square)")

BOOT = r"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
plt.rcParams["figure.dpi"] = 80
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from overlay import data, strategy as st
RED, GREY, BLUE, GREEN, AMBER = "#c0392b", "#8b949e", "#1f6feb", "#2ea44f", "#dab617"
"""


def md(text):
    return new_markdown_cell(text.strip("\n").rstrip())


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


VERDICT_TEXT = f"""
**Signal: {REAL['signal']}.** The pre-registered monthly forecaster (24-month match, 5
analogues, 12-month forecast, 1950-2018 out of sample) has a Newey-West t of **{REAL['m_t']}**
and an out-of-sample R² of {REAL['m_r2']} against the plain historical drift; the daily S&P 500
version gives t **{REAL['d_t']}**. Across all **{REAL['n_combos']} combinations** tried,
{REAL['n_raw_sig']} ({REAL['share_raw_sig']}) were raw-significant — chance gives 5% — and the
best Holm-adjusted p is **{REAL['holm_min_p']}**. The one number that survives a two-sided
correction points the *wrong* way ({REAL['worst']}, t {REAL['worst_t']}).

**Tradability: {REAL['trad']}.** Going to cash when the analogues forecast a loss (one period
of lag, 10 bp one-way): net excess Sharpe **{REAL['m_sharpe']} vs {REAL['m_bh']}** for
buy-and-hold monthly, **{REAL['d_sharpe']} vs {REAL['d_bh']}** daily; White's Reality Check over
every rule, p = {REAL['rc_m']} and {REAL['rc_d']}.

**Shape check: {REAL['myth']}.** Real history matches itself a shade better than a random walk
(median best match {REAL['real_match']} vs {REAL['null_match']}), mostly because of volatility
clustering ({REAL['volpath_match']} with the real volatility path, {REAL['block_match']} for a
6-month block bootstrap). The shapes rhyme; what follows them does not.
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(f"""
# The 1929 Overlay 📉
### Today's chart looks exactly like 1929. So does a coin flip's.

{SIGNAL_BADGE}
{TRAD_BADGE}
{MYTH_BADGE}

Every few years the same picture does the rounds: this year's stock market drawn on top of the
months before the 1929 crash, the two lines moving together almost perfectly, a correlation of
0.9-something in the corner. Nobody has to say what comes next.

> 📓 **This is the plain-language layer.** The full sweep, the corrections for having looked
> 126 times and the richer random-walk nulls are in
> **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. Numbers quoted in the text come from
> the fingerprinted run in [`docs/results.md`](../docs/results.md).
>
> ⚠️ **Not investment advice.** Every chart below is generated by the code beside it.
"""),
        code(BOOT),
        md(f"""
## The answer first 🎯

| Question | Answer |
|---|---|
| Does today's chart often look like 1929? | Yes — **{REAL['tmpl_share_09']}** of months since 1931 match the 1927-29 run-up at 0.9 or better. |
| Is a 0.9 match rare? | No. Search the past of a pure random walk and you find one **{REAL['rw_share_09']}** the time. |
| Did a crash follow the 1929 look-alikes? | Barely more often than usual: {REAL['crash_match']} vs {REAL['crash_all']} fell 20% within a year. |
| Does forecasting from look-alikes work? | No: no skill out of sample on 69 years of monthly or 23 years of daily data. |
| Can you trade it? | No: it trails simply holding the market on both tapes. |
"""),
        md("""
## 1 · The claim

Here is the chart, built from real data: the US stock market (total return, dividends
reinvested) over the 24 months to January 2014, laid over the 24 months to August 1929. An
overlay of exactly this kind circulated widely in early 2014.
"""),
        code(r"""
m = data.load_monthly()          # REAL tape: Fama-French US market, monthly TOTAL return
lp = m["logp"]
def window(end, n=24, after=0):
    i = lp.index.get_loc(pd.Timestamp(end))
    return lp.iloc[i - n + 1: i + 1 + after]
a = window("1929-08-31", after=12)
b = window("2014-01-31", after=12)
za = (a.iloc[:24] - a.iloc[:24].mean()) / a.iloc[:24].std()
zb = (b.iloc[:24] - b.iloc[:24].mean()) / b.iloc[:24].std()
corr = np.corrcoef(a.iloc[:24], b.iloc[:24])[0, 1]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(-23, 13)
ax.plot(x, (a - a.iloc[:24].mean()) / a.iloc[:24].std(), color=RED, lw=2.5,
        label="1927-09 → 1930-08 (the 1929 crash)")
ax.plot(x, (b - b.iloc[:24].mean()) / b.iloc[:24].std(), color=BLUE, lw=2.5,
        label="2012-02 → 2015-01")
ax.axvline(0, color="k", lw=1.2, ls="--")
ax.axvspan(0, 12, color=GREY, alpha=0.15)
ax.text(0.5, ax.get_ylim()[1] * 0.85, "what came NEXT", fontsize=10)
ax.set_xlabel("months relative to the overlay date"); ax.set_ylabel("rescaled log index")
ax.set_title(f"24-month overlay: correlation {corr:.3f}", fontsize=11)
ax.legend(fontsize=9); plt.show()
print(f"next 12 months after Aug 1929: {np.expm1(a.iloc[-1] - a.iloc[23]):+.1%}")
print(f"next 12 months after Jan 2014: {np.expm1(b.iloc[-1] - b.iloc[23]):+.1%}")
"""),
        md("""
A correlation of 0.95 sounds like proof. The question this notebook keeps asking is the one the
chart never does: **compared to what?**

## 2 · So what?

If the shape of the recent past told you what comes next, it would be the most valuable
indicator in finance: a free crash detector, from nothing but a price chart. It would also mean
markets repeat themselves in a way that thousands of professional investors somehow fail to
exploit. Either claim deserves a test before it deserves a reaction.

## 3 · How we'd know

We turn the chart into a machine that can be scored. **At every month**, it looks only at the
past, finds the five stretches of history whose shape best matches the last two years, and
predicts that the next year will do what followed them, on average. Then we compare those
predictions with what actually happened, from 1950 to 2018, and do the same on daily S&P 500
data from 2000 to 2022.

What would make us say **mirage**, written down before running it: the predictions do no better
than "the market drifts up at its usual rate", the best of the many variants tried stops looking
special once you account for how many were tried, and a pure random walk produces matches just
as impressive as the real ones.

> 🔬 **For the quants.** Newey-West t-statistics for the overlapping 12-month outcomes, Holm
> across 126 combinations, White's Reality Check across every timing rule. Details in notebook 02.

## 4 · The teardown

### 4.1 Two charts that have nothing to do with each other
"""),
        code(r"""
mo = data.tape_moments(m["lr"], 12)   # the REAL tape's drift and volatility
rng = np.random.default_rng(29)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
for ax in axes.ravel():
    p1 = np.cumsum(mo["mu"] + mo["sigma"] * rng.standard_normal(24))
    p2 = np.cumsum(mo["mu"] + mo["sigma"] * rng.standard_normal(24))
    c = np.corrcoef(p1, p2)[0, 1]
    ax.plot((p1 - p1.mean()) / p1.std(), color=BLUE, lw=2)
    ax.plot((p2 - p2.mean()) / p2.std(), color=RED, lw=2)
    ax.set_title(f"correlation {c:+.2f}", fontsize=10)
fig.suptitle("six pairs of INDEPENDENT random walks (same drift and vol as the market)",
             fontsize=11)
plt.tight_layout(); plt.show()
"""),
        md("""
Every pair above is two coin-flipping machines that never saw each other. Yet some of them
"track" convincingly. The reason is simple: **anything that trends correlates with anything else
that trends.** Two lines that both go up are correlated, whatever made them go up.
"""),
        code(r"""
p = st.pair_correlation_null(24, mo["mu"], mo["sigma"], n_sims=20000)
fig, ax = plt.subplots(figsize=(10, 4.8))
bins = np.linspace(-1, 1, 81)
ax.hist(p["price"], bins=bins, color=RED, alpha=0.75, label="correlation of the PRICE PATHS")
ax.hist(p["returns"], bins=bins, color=BLUE, alpha=0.6,
        label="correlation of the monthly RETURNS (same pairs)")
ax.set_xlabel("correlation between two independent random walks, 24 months")
ax.set_ylabel("count of 20,000 pairs"); ax.legend(fontsize=9); plt.show()
print(f"price paths above 0.7: {p['p_price_gt_07']:.1%}   above 0.9: {p['p_price_gt_09']:.1%}")
print(f"returns above 0.5:     {p['p_returns_gt_05']:.2%}")
"""),
        md("""
The red spread is what overlay charts measure. The blue spike is what actually carries
information — the month-by-month moves — and there, unrelated series look unrelated.

### 4.2 Now let the chartist *search*

One pair rarely hits 0.9. But the overlay chart is never one pair: someone looked through a
century of history and picked the stretch that fits best. Do that to a random walk:
"""),
        code(r"""
rw = st.random_walk_paths(1, len(m), mo["mu"], mo["sigma"], seed=7)[0]
s = st.analogue_search(rw, 24, [len(rw) - 1], k_max=1)
e = int(s["ends"][0, 0]); c = s["corr"][0, 0]
q = rw[-24:]; h_ = rw[e - 23: e + 13]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(np.arange(-23, 1), (q - q.mean()) / q.std(), color=BLUE, lw=2.5,
        label="a random walk: its last 24 'months'")
ax.plot(np.arange(-23, 13), (h_ - h_[:24].mean()) / h_[:24].std(), color=RED, lw=2.5,
        label="its best match in its own past ... and what followed")
ax.axvline(0, color="k", ls="--", lw=1.2)
ax.set_title(f"a ghost from a coin flip: best-match correlation {c:.3f}", fontsize=11)
ax.legend(fontsize=9); plt.show()
b = st.best_match_null(len(m), 24, mo["mu"], mo["sigma"], n_sims=300)
print(f"searching 92 years of a random walk: median best match {np.median(b):.3f}; "
      f"above 0.9 in {np.mean(b > 0.9):.0%} of walks")
"""),
        md(f"""
Searching the past of a pure random walk finds a match above 0.9 **{REAL['rw_share_09']}** the
time. A 0.9 on an overlay chart is not a discovery; it is what searching produces.

### 4.3 The real market's matches against a random walk's
"""),
        code(r"""
idx = st.eval_positions(m.index, "1950-01-31")
real = st.analogue_search(m["logp"].to_numpy(), 24, idx, k_max=1)["corr"][:, 0]
null = st.null_search_distribution(len(m), 24, mo["mu"], mo["sigma"], idx, n_sims=8)
fig, ax = plt.subplots(figsize=(10, 4.8))
bins = np.linspace(0.6, 1.0, 41)
ax.hist(null.ravel(), bins=bins, density=True, color=GREY, alpha=0.7,
        label="random walks, same drift and vol, same search")
ax.hist(real, bins=bins, density=True, color=RED, alpha=0.6, label="the real market, 1950-2018")
ax.set_xlabel("best-match correlation found each month"); ax.legend(fontsize=9); plt.show()
print(f"median best match: real {np.median(real):.3f}, random walk {np.median(null):.3f}")
"""),
        md(f"""
The real market's matches are a shade tighter than a plain random walk's — and in notebook 02
most of that gap disappears once the random walk is allowed to be calm and stormy at the same
times the real market was. Either way, both sit up near 0.9: impressive-looking matches are the
normal state of affairs.

> 🔬 **For the quants.** Real {REAL['real_match']}, Gaussian walk {REAL['null_match']}, a walk
> with the real volatility path {REAL['volpath_match']}, a 6-month block bootstrap of the real
> returns {REAL['block_match']}.

### 4.4 The 1929 chart, scored month by month since 1931
"""),
        code(r"""
ov = st.template_overlay(m["logp"], "1929-08-31", 24, 12)
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(ov.index, ov["corr"], color=BLUE, lw=1)
axes[0].axhline(0.9, color=RED, ls="--", lw=1.5, label="0.9 — 'looks just like 1929'")
axes[0].set_ylabel("corr with the 1927-29 run-up"); axes[0].legend(fontsize=9, loc="lower left")
hi = ov["corr"] >= 0.9
axes[1].bar(ov.index[~hi], np.expm1(ov["fwd"][~hi]) * 100, width=31, color=GREY)
axes[1].bar(ov.index[hi], np.expm1(ov["fwd"][hi]) * 100, width=31, color=RED,
            label="months that 'looked like 1929'")
axes[1].axhline(-20, color="k", ls=":", lw=1.2)
axes[1].set_ylabel("next 12 months, %"); axes[1].legend(fontsize=9, loc="lower left")
plt.tight_layout(); plt.show()
for th in (0.9, 0.95):
    e = st.overlay_episodes(ov, th)
    print(f"match >= {th}: {e['share_match']:.0%} of months | fell 20%+ within a year: "
          f"{e['crash_rate_match']:.1%} (vs {e['crash_rate_all']:.1%} for all months)")
"""),
        md(f"""
"Looks like 1929" is a description of **{REAL['tmpl_share_09']}** of all months since 1931 —
any strong two-year rally does. The red bars sit above the dotted −20% line almost exactly as
often as the grey ones. Those months did go on to earn less than average
({REAL['fwd_match']} vs {REAL['fwd_all']} a year), but that is mostly "after a big rally, returns
cool off", not a crash: once the size of the rally is accounted for, the 1929 shape adds little.

### 4.5 The forecasting machine, scored
"""),
        code(r"""
H = st.HEADLINE_MONTHLY
logp = m["logp"].to_numpy()
s = st.analogue_search(logp, H["L"], idx, k_max=H["k"], gap=H["L"])
fc = st.analogue_forecast(logp, s, H["h"], H["k"])
y = st.realised_forward(logp, idx, H["h"])
ev = st.evaluate_forecast(fc, y, st.historical_drift_forecast(logp, idx, H["h"]), H["h"])
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(fc * 100, y * 100, s=10, alpha=0.4, color=BLUE)
ax.axhline(0, color="k", lw=1); ax.axvline(0, color="k", lw=1)
ax.set_xlabel("what the look-alikes predicted for the next 12 months, %")
ax.set_ylabel("what actually happened, %")
ax.set_title(f"monthly, 1950-2018: correlation {ev['corr']:+.2f}, HAC t {ev['t']:+.2f}",
             fontsize=11)
plt.show()
"""),
        md(f"""
A cloud. The predictions explain essentially nothing about what followed, and they are worse
than simply assuming the market's long-run average (out-of-sample R² {REAL['m_r2']}). The one
variant out of {REAL['n_combos']} that looked best is no better than what you would expect to
find by chance among that many tries.

### 4.6 The control: when the past *does* repeat, the machine sees it
"""),
        code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, s_, ttl in ((axes[0], 1.0, "a market where a pattern TRULY repeats"),
                    (axes[1], 0.0, "the same market without it (pure random walk)")):
    tape, truth = data.synthetic_tape(signal_strength=s_, seed=1015)  # SYNTHETIC
    lp_ = tape["logp"].to_numpy()
    ix = st.eval_positions(tape.index, "1950-01-31")
    ss = st.analogue_search(lp_, 24, ix, k_max=5, gap=24)
    f_ = st.analogue_forecast(lp_, ss, 12, 5); y_ = st.realised_forward(lp_, ix, 12)
    e_ = st.evaluate_forecast(f_, y_, st.historical_drift_forecast(lp_, ix, 12), 12)
    ax.scatter(f_ * 100, y_ * 100, s=8, alpha=0.4, color=GREEN if s_ else GREY)
    ax.set_title(f"{ttl}\ncorr {e_['corr']:+.2f}, t {e_['t']:+.2f}", fontsize=10)
    ax.set_xlabel("predicted next 12 months, %")
axes[0].set_ylabel("actual next 12 months, %")
plt.tight_layout(); plt.show()
"""),
        md("""
In a synthetic market where a distinctive shape really is always followed by the same fall, the
same machine finds it easily. So the method is not broken. The real market simply does not
contain what the chart implies it does.

## 5 · The verdict
""" + VERDICT_TEXT),
        md(f"""
## 6 · Could you trade it?

The simplest trade: be in the market when the look-alikes predict gains, in cash when they
predict losses, acting a month after the signal and paying a small cost each switch. Over
1950-2018 it earned a risk-adjusted return (Sharpe ratio, over cash) of **{REAL['m_sharpe']}**,
against **{REAL['m_bh']}** for simply staying invested; on daily data since 2000,
**{REAL['d_sharpe']}** against **{REAL['d_bh']}**. Testing every variant at once (White's Reality
Check) gives p = {REAL['rc_m']}: none of them beats doing nothing.

## 7 · Going further 🚪

- **Next time you see the overlay**, ask for two numbers it never shows: how often today's chart
  matches *any* old rally that well, and what followed all of those matches, not just the one
  that crashed.
- **The shape is mostly the size of the rally.** A strong two-year run matches 1929 because 1929
  was a strong two-year run. Whether returns cool after big rallies is a real, old and contested
  question — a different study from this one.
- **One odd corner of the grid predicts backwards** on daily data (notebook 02, §7). It came out
  of a search, so it is a lead for someone to test on fresh data, not a strategy.
- Fork it: swap in a different distance (dynamic time warping, Euclidean on scaled paths), a
  different template (1987, 2000) or another market. The random-walk null in `overlay.strategy`
  works for any of them.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(f"""
# The 1929 Overlay — a quantitative teardown 🔬
### Analogue forecasting · spurious path correlation · 126 combinations, corrected

{SIGNAL_BADGE}
{TRAD_BADGE}
{MYTH_BADGE}

The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The study
generalises the viral overlay chart into a k-nearest-analogue forecaster, scores it out of sample
on two real tapes with overlap-robust inference, reports the whole parameter grid with Holm and
Reality-Check corrections, and calibrates the famous-looking matches against random walks — plain,
volatility-path-matched and block-bootstrapped.

> ⚠️ **Not investment advice.** Tapes: Fama-French US market **monthly total return**
> (1926-07 → 2018-11, from `arch`) and S&P 500 **daily price index** (1990-01 → 2022-11, from
> `skfolio`), both via `quantlab.bundled`, SHA-256 pinned. Headline numbers mirror
> [`docs/results.md`](../docs/results.md).
>
> 💡 **The `💡 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT),
        code(r"""
m = data.load_monthly()   # REAL: monthly TOTAL return, nominal, with T-bill rf
d = data.load_daily()     # REAL: daily PRICE index, nominal, rf = 0 (none ships)
mm, dm = data.tape_moments(m["lr"], 12), data.tape_moments(d["lr"], 252)
print(f"monthly {m.index[0].date()}..{m.index[-1].date()}  n={len(m)}  fp {data.fingerprint(m[['ret','rf']])}"
      f"  drift {mm['mu_ann']:.2%}/yr vol {mm['sigma_ann']:.2%}/yr")
print(f"daily   {d.index[0].date()}..{d.index[-1].date()}  n={len(d)}  fp {data.fingerprint(d[['px']])}"
      f"  drift {dm['mu_ann']:.2%}/yr vol {dm['sigma_ann']:.2%}/yr")
"""),
        md("## Verdict, up front\n" + VERDICT_TEXT + """
> 💡 **In plain words.** Look-alike charts are cheap, and what followed the look-alikes tells
> you nothing about what follows today.
"""),
        md("""
## 1 · Hypotheses (pre-registered)

- **H₁ (skill).** The analogue forecast's slope on the realised h-period return is positive:
  Newey-West t ≥ 2 on the headline spec of **both** tapes (monthly L=24 m, h=12 m, k=5; daily
  L=250 d, h=63 d, k=5; price-path matching), and the best combination survives Holm across the
  whole grid.
- **H₂ (tradability).** A long/flat rule (one period of execution lag, 10 bp one-way) beats
  buy-and-hold on net excess Sharpe on both tapes, and White's Reality Check rejects.
- **H₃ (spurious matching).** The real tape's best-match correlations are no higher than those of
  random walks with the same drift and vol, searched identically.
- **H₄ (power).** On a synthetic tape with a genuinely recurring template, the same test rejects;
  on the matched null it rejects at about its nominal rate.

The stamps follow `strategy.verdict` / `strategy.myth_check`, thresholds fixed before the real
run and unit-tested in both directions.

## 2 · The machinery

At each evaluation date `t`: z-normalise the last `L` points (log-price path, or L log returns
for `zret`), dot it with every past window ending at `e ≤ t − max(L, h)` — a Pearson
correlation — then pick `k` analogues greedily, removing every window within `L` of each pick so
the ten best are not ten one-period shifts of one episode. The forecast is the mean of
`logp[e+h] − logp[e]` over the picks.
"""),
        code(r"""
idx_m = st.eval_positions(m.index, "1950-01-31")
s = st.analogue_search(m["logp"].to_numpy(), 24, idx_m, k_max=10, gap=24)
t_last = len(idx_m) - 1
ends, cors = s["ends"][t_last], s["corr"][t_last]
print(f"query: the 24 months to {m.index[idx_m[t_last]].date()}")
print(pd.DataFrame({"analogue ends": m.index[ends].date, "corr": cors}).to_string(index=False))
gaps = np.diff(np.sort(ends))
print(f"\nminimum spacing between analogues: {gaps.min()} months (>= L = 24 by construction)")
"""),
        md("""
> 💡 **In plain words.** Ten past look-alikes for November 2018, all above 0.9 — and none of
> them knows anything about December.

## 3 · The spurious-correlation core

### 3.1 One pair: the distribution of corr(price paths) for independent walks, by window
"""),
        code(r"""
rows = []
for L in (12, 24, 36, 60, 120, 250):
    mo = mm if L <= 120 else dm
    p = st.pair_correlation_null(L, mo["mu"], mo["sigma"], n_sims=8000)
    rows.append({"L": L, "tape calib": "monthly" if L <= 120 else "daily",
                 "sd(price corr)": np.std(p["price"]), "P(price>0.7)": p["p_price_gt_07"],
                 "P(price>0.9)": p["p_price_gt_09"], "sd(return corr)": np.std(p["returns"]),
                 "P(ret>0.5)": p["p_returns_gt_05"]})
t3 = pd.DataFrame(rows).set_index("L")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t3.index, t3["sd(price corr)"], "o-", color=RED, lw=2, label="sd of PRICE-path corr")
ax.plot(t3.index, t3["sd(return corr)"], "s-", color=BLUE, lw=2, label="sd of RETURN corr")
ax.set_xscale("log"); ax.set_xlabel("window length L"); ax.legend(fontsize=9); plt.show()
t3
"""),
        md("""
The return correlation shrinks like `1/√L`, as it should. The price-path correlation does
**not**: its spread barely moves with the window length (Phillips 1986 — it converges to a
non-degenerate random variable). Longer overlays are no safer.

### 3.2 The search: best match against breadth of history
"""),
        code(r"""
rows = []
for n_hist in (120, 300, 600, 1100):
    b = st.best_match_null(n_hist, 24, mm["mu"], mm["sigma"], n_sims=200, seed=n_hist)
    z = st.best_match_null(n_hist, 24, mm["mu"], mm["sigma"], metric="zret", n_sims=200,
                           seed=n_hist)
    rows.append({"months searched": n_hist, "median best (price)": np.median(b),
                 "P(best>0.9)": np.mean(b > 0.9), "median best (zret)": np.median(z)})
pd.DataFrame(rows).set_index("months searched")
"""),
        md("""
> 💡 **In plain words.** The longer the history you are allowed to rummage through, the better
> the best "match" gets — for a pure random walk. Ninety years of history buys you a 0.9 about
> half the time.

### 3.3 Like for like, with richer nulls

Same dates, same search. Three nulls: a Gaussian walk with the tape's drift and vol; a walk with
the **real volatility path** (direction coin-flipped); a **6-month stationary block bootstrap**
of the real returns (short memory and fat tails kept, recurrence across years destroyed; blocks
much shorter than L so no window can be copied).
"""),
        code(r"""
lp_m = m["logp"].to_numpy()
real = st.analogue_search(lp_m, 24, idx_m, k_max=1)["corr"][:, 0]
nulls = {
    "Gaussian walk": st.null_search_distribution(len(m), 24, mm["mu"], mm["sigma"], idx_m,
                                                 n_sims=8),
    "real vol path": st.null_search_paths(st.vol_path_walks(m["lr"].to_numpy(), 8), 24, idx_m),
    "6m block bootstrap": st.null_search_paths(
        st.block_bootstrap_walks(m["lr"].to_numpy(), 8, mean_block=6.0), 24, idx_m),
}
fig, ax = plt.subplots(figsize=(10, 4.8))
bins = np.linspace(0.6, 1.0, 41)
for (k_, v_), c_ in zip(nulls.items(), (GREY, AMBER, GREEN)):
    ax.hist(v_.ravel(), bins=bins, density=True, histtype="step", lw=2, color=c_, label=k_)
ax.hist(real, bins=bins, density=True, color=RED, alpha=0.45, label="REAL monthly tape")
ax.set_xlabel("best-match correlation (L = 24 months)"); ax.legend(fontsize=9); plt.show()
pd.DataFrame({"median": [np.median(real)] + [np.median(v) for v in nulls.values()],
              "share > 0.9": [np.mean(real > 0.9)] + [np.mean(v > 0.9) for v in nulls.values()]},
             index=["REAL"] + list(nulls))
"""),
        md("""
> 💡 **In plain words.** The real market rhymes with itself a little more than a coin-flip walk,
> and nearly all of that is because turbulent years cluster (a window dominated by one crash or
> one boom looks like a straight line, and straight lines match each other). Put the clustering
> back and the real tape looks like the nulls. The Shape-check stamp is "Confirmed" by its
> pre-registered rule (real beats the *Gaussian* walk), and that is the whole of what it confirms.

### 3.4 The 1929 template — and why it is the trailing rally in disguise
"""),
        code(r"""
ov = st.template_overlay(m["logp"], "1929-08-31", 24, 12).dropna()
trail = (m["logp"] - m["logp"].shift(24)).reindex(ov.index)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].scatter(trail * 100, ov["corr"], s=6, alpha=0.4, color=BLUE)
axes[0].set_xlabel("trailing 24-month log return, %"); axes[0].set_ylabel("corr with 1927-29")
axes[0].set_title(f"corr(match, trailing return) = {np.corrcoef(ov['corr'], trail)[0,1]:.2f}",
                  fontsize=10)
ind = (ov["corr"] >= 0.9).to_numpy(float)
res = ind - np.polyval(np.polyfit(trail, ind, 1), trail)
rows = {"match >= 0.9": st.ols_hac(ov["fwd"].to_numpy(), ind, 12)["t_b"],
        "trailing return": st.ols_hac(ov["fwd"].to_numpy(), trail.to_numpy(), 12)["t_b"],
        "match, orthogonal to trailing": st.ols_hac(ov["fwd"].to_numpy(), res, 12)["t_b"]}
axes[1].barh(list(rows), list(rows.values()), color=[RED, GREY, AMBER])
axes[1].axvline(-2, color="k", ls=":"); axes[1].axvline(2, color="k", ls=":")
axes[1].set_xlabel("HAC t (12 lags) on next-12-month log return"); plt.tight_layout(); plt.show()
pos = m.index.get_loc(pd.Timestamp("1929-08-31"))
tn = st.template_null_share(m["logp"].to_numpy()[pos - 23: pos + 1], mm["mu"], mm["sigma"])
pd.DataFrame([{**st.overlay_episodes(ov, th), "random walk share": tn[th]}
              for th in (0.8, 0.9, 0.95)]).set_index("threshold")[
    ["n_match", "share_match", "random walk share", "fwd_match", "fwd_all", "diff_t",
     "crash_rate_match", "crash_rate_all"]]
"""),
        md("""
> 💡 **In plain words.** "Looks like 1929" mostly means "went up a lot for two years". Months
> like that did earn less over the following year — an old, contested long-horizon reversal story
> — but they did not crash more often, and the 1929 *shape* adds little once the size of the rally
> is known.

## 4 · The headline forecasters
"""),
        code(r"""
idx_d = st.eval_positions(d.index, "2000-01-03", 5)
out = {}
for tag, tape, ix, H, step in (("monthly", m, idx_m, st.HEADLINE_MONTHLY, 1),
                               ("daily", d, idx_d, st.HEADLINE_DAILY, 5)):
    lp = tape["logp"].to_numpy()
    s_ = st.analogue_search(lp, H["L"], ix, k_max=H["k"], gap=max(H["L"], H["h"]))
    fc = st.analogue_forecast(lp, s_, H["h"], H["k"])
    y = st.realised_forward(lp, ix, H["h"]); b = st.historical_drift_forecast(lp, ix, H["h"])
    base = int(np.ceil(H["h"] / step))
    row = st.evaluate_forecast(fc, y, b, base)
    row.update({f"t ({mult}x lags)": st.evaluate_forecast(fc, y, b, base * mult)["t"]
                for mult in (2, 4)})
    row["corr 95% CI"] = st.stationary_bootstrap_corr(fc, y, 300, 2 * base)
    out[tag] = row
    out[tag + "_series"] = (tape.index[ix], fc, y)
pd.DataFrame({k: v for k, v in out.items() if not k.endswith("_series")}).T[
    ["n", "corr", "corr 95% CI", "slope", "t", "t (2x lags)", "t (4x lags)", "hit", "hit_t",
     "hit_raw", "up_share", "oos_r2"]]
"""),
        code(r"""
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
for ax, tag in zip(axes, ("monthly", "daily")):
    dt_, fc, y = out[tag + "_series"]
    ax.plot(dt_, y * 100, color=GREY, lw=1, label="realised")
    ax.plot(dt_, fc * 100, color=RED, lw=1.5, label="analogue forecast")
    ax.set_ylabel(f"{tag}: h-period log return, %"); ax.legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.show()
"""),
        md("""
> 💡 **In plain words.** The forecast (red) wanders around a few percent; reality (grey) swings
> by tens of percent and pays it no attention. Note the raw sign hit rate looks respectable only
> because it roughly equals the share of up periods — a forecast that is "mostly positive" in a
> market that mostly rises. Against the drift, the hit rate is a coin.

## 5 · The full sweep — 126 combinations, all reported
"""),
        code(r"""
sm = st.sweep(m, st.GRID_MONTHLY, 12, "1950-01-31", step=1)
sd = st.sweep(d, st.GRID_DAILY, 252, "2000-01-03", step=5)
tm, td = sm["table"].assign(tape="monthly"), sd["table"].assign(tape="daily")
allt = pd.concat([tm, td], ignore_index=True)
allt["p_holm"] = st.holm(allt["p"].to_numpy())
allt["p_two"] = 2 * np.minimum(allt["p"], 1 - allt["p"])
allt["p_holm_two"] = st.holm(allt["p_two"].to_numpy())
print(f"{len(allt)} combinations; raw one-sided p<0.05: {(allt['p'] < 0.05).sum()} "
      f"({(allt['p'] < 0.05).mean():.1%}); min Holm p {allt['p_holm'].min():.2f}; "
      f"two-sided Holm survivors: {(allt['p_holm_two'] < 0.05).sum()}")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
xs = np.linspace(-5, 5, 200)
axes[0].hist(allt["t"], bins=30, density=True, color=BLUE, alpha=0.7, label="126 HAC t-stats")
axes[0].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color="k", lw=2, label="N(0,1)")
axes[0].set_xlabel("HAC t of forecast slope"); axes[0].legend(fontsize=8)
for (tape, metric), g in allt.groupby(["tape", "metric"]):
    axes[1].scatter(g["L"] + (5 if metric == "zret" else 0), g["t"], s=25,
                    label=f"{tape} {metric}", alpha=0.8)
axes[1].axhline(2, color="k", ls=":"); axes[1].axhline(-2, color="k", ls=":")
axes[1].set_xscale("log"); axes[1].set_xlabel("window L"); axes[1].set_ylabel("HAC t")
axes[1].legend(fontsize=7); plt.tight_layout(); plt.show()
"""),
        code(r"""
cols = ["tape", "metric", "L", "h", "k", "corr", "t", "p", "p_holm", "hit", "oos_r2",
        "sharpe_net", "sharpe_bh", "alpha_t", "time_in_market"]
print(allt[cols].sort_values("t", ascending=False).round(3).to_string(index=False))
"""),
        md("""
> 💡 **In plain words.** The 126 t-statistics look like a draw from a standard normal — what
> you would see if nothing were there — with a few long tails, and the longest one points the
> wrong way.

## 6 · The timing rule and the Reality Check
"""),
        code(r"""
km = st.timing_rule(m, idx_m, out["monthly_series"][1], 12)
kd = st.timing_rule(d, idx_d, out["daily_series"][1], 252)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, r, ttl in ((axes[0], km, "monthly TR, cash = T-bill"),
                   (axes[1], kd, "daily price, cash = 0")):
    ax.plot(r["equity_bh"], color=GREY, lw=2, label=f"buy & hold (Sharpe {r['sharpe_bh']:.2f})")
    ax.plot(r["equity"], color=RED, lw=1.8,
            label=f"analogue long/flat, net (Sharpe {r['sharpe_net']:.2f})")
    ax.set_yscale("log"); ax.set_title(ttl, fontsize=10); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
rc_m = st.reality_check(sm["diffs"], 12, n_boot=300)
rc_d = st.reality_check(sd["diffs"], 252, n_boot=300)
print(f"rules beating B&H on net Sharpe: monthly {(tm['sharpe_net'] > tm['sharpe_bh']).sum()}/"
      f"{len(tm)}, daily {(td['sharpe_net'] > td['sharpe_bh']).sum()}/{len(td)}")
rv_m = st.reality_check(sm["diffs_vm"], 12, n_boot=300)
rv_d = st.reality_check(sd["diffs_vm"], 252, n_boot=300)
print(f"Reality Check, return race: monthly p {rc_m['reality_check_pvalue']:.3f} "
      f"(best {rc_m['best_rule']}), daily {rc_d['reality_check_pvalue']:.3f}")
print(f"Reality Check, vol-matched Sharpe race: monthly p {rv_m['reality_check_pvalue']:.3f} "
      f"(best {rv_m['best_rule']}), daily {rv_d['reality_check_pvalue']:.3f}")
pd.DataFrame({"monthly": {k: km[k] for k in ("sharpe_gross", "sharpe_net", "sharpe_bh",
                                            "alpha_ann", "alpha_t", "time_in_market",
                                            "turnover_per_year", "max_dd", "max_dd_bh")},
              "daily": {k: kd[k] for k in ("sharpe_gross", "sharpe_net", "sharpe_bh",
                                          "alpha_ann", "alpha_t", "time_in_market",
                                          "turnover_per_year", "max_dd", "max_dd_bh")}})
"""),
        md("""
**Timing, exactly:** the forecast made at the close of period `t` is executed at the close of
`t + 1` and first earns period `t + 2`; costs 10 bp one-way × traded NAV; both legs' Sharpe
ratios are in excess of cash. The Reality Check is run as a *return race* (rule minus
buy-and-hold) and as a *Sharpe race* (each rule first scaled to buy-and-hold's volatility, so
de-risking gets credit); here with 300 resamples for speed, 1,000 in `docs/results.md`.

> 💡 **In plain words.** The rule mostly sits in the market anyway, steps out at random moments,
> and pays for the privilege. The best of every variant tried still does not beat doing nothing.

## 7 · The wrong-way lead
"""),
        code(r"""
lp = d["logp"].to_numpy()
s_ = st.analogue_search(lp, 63, idx_d, k_max=5, gap=63)
fc = st.analogue_forecast(lp, s_, 63, 5); y = st.realised_forward(lp, idx_d, 63)
ok = np.isfinite(y)
roll = pd.Series(fc[ok], index=d.index[idx_d][ok]).rolling(150).corr(
    pd.Series(y[ok], index=d.index[idx_d][ok]))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(roll, color=RED, lw=1.8); ax.axhline(0, color="k", lw=1)
ax.set_ylabel("rolling corr (150 weekly forecasts)")
ax.set_title("daily price-path analogues, L=63, h=63, k=5: forecast vs realised", fontsize=10)
plt.show()
"""),
        md(f"""
The most extreme number in the grid ({REAL['worst']}, t {REAL['worst_t']}, two-sided Holm p
{REAL['worst_holm2']}) says the short-window daily analogues **anti-forecast** the next quarter.
It is robust to the lag choice, present in both halves and not the trailing return in disguise
(`docs/results.md` §7) — but it is one corner of a searched grid, absent from the monthly tape,
and a contrarian rule built on it after seeing the sign does not beat buy-and-hold net of costs.
Recorded as a lead for fresh data, not a finding.

## 8 · Power: the detector on a template that truly recurs (SYNTHETIC)
"""),
        code(r"""
rows = []
for sstr in (0.0, 0.25, 0.5, 0.75, 1.0):
    res = st.null_sweep_power(
        lambda sd_, s=sstr: data.synthetic_tape(n_periods=len(m), signal_strength=s,
                                                mu=mm["mu"], sigma=mm["sigma"], seed=sd_)[0],
        None, 12, "1950-01-31", range(1015, 1025), st.HEADLINE_MONTHLY)
    rows.append({"signal_strength": sstr, "reject (t>=2)": (res["t"] >= 2).mean(),
                 "median t": res["t"].median(), "median best match": res["match_corr_median"].median()})
pw = pd.DataFrame(rows).set_index("signal_strength")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(pw.index, pw["reject (t>=2)"] * 100, "o-", color=GREEN, lw=2.5)
ax.axhline(2.3, color="k", ls=":", label="nominal rate of t >= 2 under the null (2.3%)")
ax.set_xlabel("signal_strength of the planted template"); ax.set_ylabel("% of tapes with t >= 2")
ax.legend(fontsize=8); plt.show()
pw
"""),
        md("""
> 💡 **In plain words.** With nothing planted the test stays quiet even though the best matches
> still look superb (~0.90). With the template planted at half strength or more, it fires. The
> machine can see a repeating pattern — the real market just does not have one.

## 9 · Could you trade it?

No. There is no forecasting signal to scale, and the costless gross Sharpe of the headline rules
already trails buy-and-hold, so capacity and impact never come into it. Execution would be
trivial (index futures or an ETF, a handful of switches a year); the problem is upstream.

## 10 · Going further

- **Other distances.** Dynamic time warping or Euclidean distance on scaled paths change the
  similarity score, not the null: any shape-matching on trending paths inherits §3.1.
- **Other templates.** 1987 and 2000 are inside the monthly tape and already part of the general
  search; a dedicated template test would mirror §3.4.
- **The reversal lead.** §3.4 (returns after big rallies) and §7 (daily anti-forecasting) both
  point at mean reversion, a separate claim with a separate literature (Fama & French 1988;
  Poterba & Summers 1988) and a strong dependence on the 1930s and on sample choice.
- **Fork the null.** `strategy.null_search_distribution`, `vol_path_walks` and
  `block_bootstrap_walks` calibrate any "today looks like X" chart in a few lines.
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
