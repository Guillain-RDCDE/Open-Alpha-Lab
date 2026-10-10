"""Notebook builder for Study 1013 — It's Official.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    cd notebooks
    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both walk the same seven desk beats (METHODOLOGY.md) at two altitudes. Every code cell reads
the **real** bundled tapes (offline, SHA-256 pinned) except the cells explicitly labelled
*synthetic*. The headline numbers quoted in prose live in ONE place — the ``REAL`` dict below,
copied from ``docs/results.md`` (regenerate both whenever ``examples/verify.py`` is re-run).
The verdict badges are the ones the real run earned, never chosen by hand.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# The real-run numbers — mirror docs/results.md (as-of 2018-11-30 monthly TR /
# 2022-11-30 daily price; fingerprints 394be76d78ac / e223c581ae22).
# --------------------------------------------------------------------------- #
REAL = {
    "signal": "Mixed", "trad": "Mirage",
    "lag_peak": "7", "lag_trough": "15",
    "trough_first": "14 of 15", "lead_median": "4", "peak_first": "12 of 15",
    "rot_share_obs": "93%", "rot_share_null": "71%", "rot_p": "0.032",
    "rot_mean_obs": "4.4", "rot_mean_null": "6.2", "rot_mean_p": "0.78", "grid_sig": "8 of 9",
    "dd_already": "−6.7%", "no_fall": "2 of 6", "share_done": "55%", "still_to_come": "−7.8%",
    "missed": "+63%", "missed_min": "+25%", "since_low": "18",
    "buy_obs": "+3.1%", "buy_null": "+8.5%", "buy_p": "0.82", "buy_p_peak": "0.43",
    "buy_p_trough": "0.93", "buy_p_daily": "0.54", "trough_lower_p": "0.07",
    "mde80": "17%", "mde80_6": "19%",
    "rule_sharpe": "0.14", "bh_sharpe": "0.52", "rule_gain": "−0.38", "rule_t": "−3.17",
    "rule_perm_p": "0.76", "rule_gain_daily": "−0.27", "rule_exposure": "24%",
    "avoid_gain": "−0.09", "avoid_gain_daily": "−0.07",
    "fp_m": "394be76d78ac", "fp_d": "e223c581ae22",
}

COLORS = {"Real": "2ea44f", "Investable": "2ea44f", "Weak": "dab617", "Mixed": "dab617",
          "Fragile": "dab617", "None": "c0392b", "Mirage": "c0392b"}


def _badges():
    s, t = REAL["signal"], REAL["trad"]
    return (f"![Signal: {s}](https://img.shields.io/badge/Signal-{s}-{COLORS[s]}"
            f"?style=flat-square)\n"
            f"![Tradability: {t}](https://img.shields.io/badge/Tradability-{t}-{COLORS[t]}"
            f"?style=flat-square)")


def F(text):
    """Fill {placeholders} from REAL without tripping over code braces."""
    for k, v in REAL.items():
        text = text.replace("{" + k + "}", v)
    return text.replace("{BADGES}", _badges())


def md(text):
    return new_markdown_cell(F(text).strip("\n").rstrip())


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


BOOT = r"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
warnings.filterwarnings("ignore", category=UserWarning)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
GREEN, RED, BLUE, GREY, AMBER = "#2ea44f", "#c0392b", "#1f6feb", "#8b949e", "#dab617"

from nberclock import data, strategy as st
m = data.load_monthly()              # Fama-French monthly, TOTAL return, 1926-07 -> 2018-11
d = data.load_daily()                # S&P 500 daily PRICE index, 1990-01 -> 2022-11
tr = data.total_return_index(m)
xs = st.excess_index(m)
cyc, anns = data.cycles(), data.announcements()
print(f"monthly TR {m.index[0].date()} -> {m.index[-1].date()}  fp {data.fingerprint(m)}")
print(f"daily price {d.index[0].date()} -> {d.index[-1].date()}  fp {data.fingerprint(d)}")
"""

SHADE = r"""
def shade_recessions(ax, lo=None, hi=None):
    for _, c in cyc.iterrows():
        if (lo is None or c.trough >= lo) and (hi is None or c.peak <= hi):
            ax.axvspan(c.peak, c.trough, color=GREY, alpha=0.18, lw=0)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(r"""
# It's Official 📣
### The NBER tells you a recession started — months after it did. Is that the moment to buy?

{BADGES}

**Real on the lead · None on the buy signal.** The stock market really does turn before the
official referee says so — but the day the referee speaks is not a special day to buy.

> 📓 **This is the plain-language layer.** The rotation tests, the power curve and the rule
> backtests are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> ⚠️ **Not investment advice.** Every chart below is drawn from the real bundled tapes by the
> code beside it; the one synthetic chart says so in its title.
"""),
        code(BOOT + SHADE),
        md(r"""
## Beat 0 · The answer first (real tape)

| Question | Answer |
|---|---|
| Does the market turn before the NBER's dates? | Yes — the market bottomed first in {trough_first} cycles since 1926. |
| Has the market "done its falling" by the peak announcement? | Sometimes. Median drawdown already in: {dd_already}; in {no_fall} there was no fall at all. |
| Has the recovery "long gone" by the trough announcement? | Yes. Median {missed} off the low. |
| Is the announcement a buy signal? | No. {buy_obs} over the next year (above bills) vs {buy_null} after random dates. |
| Could you trade it? | Every rule built on it lost to just holding. |
"""),
        md(r"""
## Beat 1 · The claim

The folk version goes like this: *the National Bureau of Economic Research is so slow that by
the time it officially declares a recession, the stock market has already done its falling — so
the announcement is a buy signal. And by the time it declares the recovery, the cheap prices are
long gone.*

It sounds plausible because the first half is famous. The committee announced that the Great
Recession had begun on **1 December 2008** — a full year after it started. It announced the end
fifteen months after the end.
"""),
        code(r"""
show = anns.assign(reference=anns.reference.dt.strftime("%Y-%m"),
                   date=anns.date.dt.strftime("%Y-%m-%d"))[["kind", "reference", "date",
                                                             "lag_months"]]
fig, ax = plt.subplots(figsize=(10, 4.5))
for kind, col, off in (("peak", RED, -0.18), ("trough", GREEN, 0.18)):
    a = anns[anns.kind == kind]
    ax.barh(np.arange(len(a)) + off, a.lag_months, height=0.34, color=col,
            label=f"{kind}: months from the turn to the announcement")
ax.set_yticks(range(6))
ax.set_yticklabels([f"{p.year} cycle" for p in anns[anns.kind == 'peak'].reference])
ax.invert_yaxis(); ax.set_xlabel("months of delay"); ax.legend(fontsize=9)
ax.set_title("How late is 'official'? (NBER announcements, 1980-2021)")
plt.show()
show
"""),
        md(r"""
Peaks get announced a median **{lag_peak} months** late; troughs a median **{lag_trough}**.
That delay is not sloppiness — the committee waits until the data are unambiguous — but it is
the raw material of the claim.

## Beat 2 · So what?

If the announcement really were a buy signal, it would be the rare market-timing rule that
needs no model, no data feed and no judgement: wait for a press release, buy. And it would say
something about markets — that investors keep selling into news everyone already knows.

## Beat 3 · How we'd know

Two separate questions, two separate tests, fixed before we looked:

1. **The lead.** Find the market's own peak and trough around each of the 16 NBER cycles since
   1926 and count how often the market turned first.
2. **The buy signal.** Compare the return over the year after each official announcement with
   the return after *random* dates. If the official dates aren't better than random ones, the
   announcement is not a signal — whatever the lead.

**"Mirage" would mean:** the market leads, the delay is real, and still nothing special happens
after the announcement.
"""),
        md(r"""
## Beat 4 · The teardown

### 4.1 A century of cycles: the market moves first
"""),
        code(r"""
ll = st.lead_lag_all(tr, d, cyc)
fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(tr.index, tr, color=BLUE, lw=1.4, label="US market, total return (log scale)")
shade_recessions(ax, hi=tr.index[-1])
okk = ll[ll.tape == "FF total return"]
ax.scatter(okk.mkt_trough, tr.reindex(okk.mkt_trough), color=GREEN, s=40, zorder=3,
           label="market's own trough")
ax.scatter(okk.mkt_peak, tr.reindex(okk.mkt_peak), color=RED, s=40, zorder=3,
           label="market's own peak")
ax.set_title("Grey = official NBER recessions. Dots = where the market actually turned")
ax.legend(fontsize=9); plt.show()
"""),
        code(r"""
fig, ax = plt.subplots(figsize=(11, 4.5))
lab = [p.strftime("%Y") for p in ll.peak]
x = np.arange(len(ll))
ax.bar(x - 0.2, ll.lead_peak, width=0.4, color=RED, label="peak: months the market topped first")
ax.bar(x + 0.2, ll.lead_trough, width=0.4, color=GREEN,
       label="trough: months the market bottomed first")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=45, fontsize=8)
ax.set_ylabel("months ahead of the NBER date"); ax.legend(fontsize=9)
plt.show()
s = st.lead_summary(ll)
print(f"market bottomed first in {s['trough_pos']} of {s['n']} uncensored cycles; "
      f"median lead {s['trough_median']:.0f} months")
"""),
        md(r"""
Almost every bar is above zero. The one big exception is 2001: the official recession ended in
November 2001, but the dot-com bear market kept going until late 2002.

> 🔬 **For the quants.** The rule that finds "the market's trough" is not neutral — on a rising
> market it tends to find a low early. Run on the same tape with the NBER calendar shifted to
> arbitrary months, it puts the market first {rot_share_null} of the time. The real calendar
> ({rot_share_obs}) still beats that (p = {rot_p}), but the *size* of the lead does not.

### 4.2 On announcement day: what was already done?
"""),
        code(r"""
at_m = st.announcement_table(tr, anns)
at_d = st.announcement_table(d, anns, bars_per_month=21)
best = st.best_available(at_m, at_d)
pk, tg = best[best.kind == "peak"], best[best.kind == "trough"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
axes[0].bar(pk.date.dt.year.astype(str), pk.already * 100, color=RED, label="fall already in")
axes[0].bar(pk.date.dt.year.astype(str), pk.still_to_come * 100, bottom=pk.already * 100,
            color=AMBER, label="fall still to come")
axes[0].set_title("Peak announcements: has the market done its falling?")
axes[0].set_ylabel("% from the market high"); axes[0].legend(fontsize=9)
axes[1].bar(tg.date.dt.year.astype(str), tg.already * 100, color=GREEN)
axes[1].set_title("Trough announcements: rebound already missed")
axes[1].set_ylabel("% above the market low")
plt.tight_layout(); plt.show()
best[["kind", "date", "lag_months", "already", "still_to_come", "fwd_12m", "tape"]]
"""),
        md(r"""
The two halves of the claim come apart here. At **trough** announcements the claim is right —
the market was already a median **{missed}** off its low. At **peak** announcements it is only
half right: in 2008 the market was down 40% when the news came, but in 1980 and 1991 it had
barely fallen, and in 2001 half the bear market was still ahead.

### 4.3 Is the announcement a buy signal?
"""),
        code(r"""
pos = st.event_positions(xs.index, anns)
r = st.rotation_test(xs, pos, 12)
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.hist(r["perm"] * 100, bins=50, color=GREY, label="same 10 dates, shifted to every other month")
ax.axvline(r["observed"] * 100, color=RED, lw=3, label="the real announcement dates")
ax.set_xlabel("average return over the next 12 months, above T-bills (%)")
ax.set_ylabel("number of shifted calendars"); ax.legend(fontsize=9)
plt.show()
print(f"after the announcements: {r['observed']:+.1%}   after random dates: "
      f"{r['null_mean']:+.1%}   share of random calendars doing at least as well: "
      f"{r['p_one_sided']:.0%}")
"""),
        md(r"""
The red line sits in the *middle-left* of the grey pile. Buying on the official dates did
worse than buying on most shifted versions of the same calendar.

### 4.4 Could twelve dates ever have told us? (synthetic)
"""),
        code(r"""
pc = st.power_curve(effects=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30), n_sims=80,
                    vol=float(m.mkt.std() * np.sqrt(12)))
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(pc.index * 100, pc.reject_rate * 100, "o-", color=BLUE, lw=2)
ax.axhline(80, color=GREY, ls="--", lw=1)
ax.set_xlabel("a buy signal of this size planted after every announcement (% over 12 months)")
ax.set_ylabel("% of synthetic worlds where the test spots it")
ax.set_title("SYNTHETIC: how big a signal 10 announcements can detect")
plt.show()
"""),
        md(r"""
With only ten announcements on the total-return tape, the test spots a planted buy signal most
of the time only once it is worth about **{mde80}** extra over a year. So "no signal" here means
"no *big* signal". It cannot rule out a small one, and neither can anyone else with this
history.

## Beat 5 · The verdict

**Signal: Mixed — Real on the lead · None on the buy signal.** The market bottomed before the
official trough in {trough_first} cycles, and that beats what the method produces on arbitrary
dates (p = {rot_p}). The announcement itself carries no detectable timing information
(p = {buy_p}).

**Tradability: Mirage.** See below.

## Beat 6 · Could you trade it?
"""),
        code(r"""
mm = m[m.index >= "1980-01-01"]
sig = {"bills while officially in recession": st.signal_avoid(mm.index, anns),
       "equity 12 m after each announcement": st.signal_buy_after(mm.index, anns, 12)}
fig, ax = plt.subplots(figsize=(11, 5))
bh = st.timing_backtest(mm.mkt, mm.rf, pd.Series(1.0, index=mm.index))
ax.semilogy((1 + bh["net_series"]).cumprod(), color=BLUE, lw=2, label="buy and hold")
for (name, s_), col in zip(sig.items(), (AMBER, RED)):
    b = st.timing_backtest(mm.mkt, mm.rf, s_, cost_bps=10)
    ax.semilogy((1 + b["net_series"]).cumprod(), color=col, lw=2, label=name)
shade_recessions(ax, lo=mm.index[0])
ax.set_title("$1 in 1980, net of 10 bp per switch, total return (log scale)")
ax.legend(fontsize=9); plt.show()
"""),
        md(r"""
Neither the believer ("buy on the news") nor the sceptic ("it's official — get out until it's
over") beats simply holding. The buyer sits in cash most of the time and misses the market's
long climb; the seller gets out after the damage and comes back after the rebound. Costs barely
matter — it is the timing that loses.

## Beat 7 · Going further 🚪

- **The trough-announcement curiosity.** The year after a "recovery is official" release was
  *below* random dates (one-tailed p ≈ {trough_lower_p}, five events). Is it a real "easy money
  is gone" effect or noise? Other countries' official cycle dating (e.g. the CEPR committee for
  the euro area) would add independent events.
- **Real-time recession probabilities** (yield curve, Sahm rule) move *before* the committee —
  see the neighbouring studies 268-sahm-rule and 626-unemployment-trend-timing.
- **Fork it:** change the windows in `nberclock.strategy.lead_lag` and watch the lead's
  p-value; it survives {grid_sig} of the window choices we tried.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(r"""
# It's Official — a quantitative teardown 🔬
### Turning-point leads vs a calendar-rotation null · an exact rotation test on 12 events · power · rules

{BADGES}

**Real on the lead · None on the buy signal.** The companion to the
[notebook for the curious](01_for_the_curious.ipynb). Inference is by exact circular-shift
(rotation) randomisation throughout, because with n = 10–12 events nothing asymptotic applies.
§3.2 is the honest null for the lead (the estimator leans), §3.4 the buy-signal test, §3.5 the
power that bounds what "not significant" means.

> ⚠️ **Not investment advice.** Monthly tape: Fama-French market **total return** and T-bills
> (`arch`, fp `{fp_m}`, as-of 2018-11-30). Daily tape: S&P 500 **price index**, no dividends
> (`skfolio` 1.8.1, fp `{fp_d}`, as-of 2022-11-30). Numbers in prose mirror
> [`docs/results.md`](../docs/results.md).
>
> 💡 **The `💡 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT + SHADE),
        md(r"""
## Verdict, up front (real tape)

| Axis | Stamp | Decisive numbers |
|---|---|---|
| **Signal** | Mixed — Real on the lead · None on the buy signal | market trough first in {trough_first} uncensored cycles; vs shifted calendars {rot_share_obs} vs {rot_share_null}, p = {rot_p}; buy signal: {buy_obs} vs {buy_null} after random dates, rotation p = {buy_p}; MDE₈₀ = {mde80} |
| **Tradability** | Mirage | "equity 12 m after each announcement": net excess Sharpe {rule_sharpe} vs {bh_sharpe} (Δ {rule_gain}, HAC t {rule_t}; shifted-date p = {rule_perm_p}); daily Δ {rule_gain_daily}; "bills while officially in recession": Δ {avoid_gain} / {avoid_gain_daily} |

> 💡 **In plain words.** The market does move first, but the press release is just a date.

## 1 · Hypotheses (fixed before the run)

- **H₁ (lead).** The market trough precedes the NBER trough in more cycles than (a) a coin and
  (b) the same estimator on month-shifted NBER calendars would produce.
- **H₂ (buy signal, primary).** Mean 12-month excess return after the 10 announcements on the
  monthly total-return tape exceeds every-circular-shift random dates, one-sided p < 0.05.
- **H₃ (tradability).** The rule "equity for 12 months after each announcement" beats
  buy-and-hold on net excess Sharpe on both tapes, with one execution lag and 10 bp one-way.

Window choices: market peak = max over [P − 24, P + 3] months; market trough = min over
[market peak, T + 12]. Horizon 12 months; 3/6/24 shown alongside.
"""),
        md(r"""
## 2 · The calendar and the delay
"""),
        code(r"""
anns.assign(reference=anns.reference.dt.strftime("%Y-%m"))
"""),
        md(r"""
## 3 · The teardown

### 3.1 Turning points, every cycle since 1926
"""),
        code(r"""
ll = st.lead_lag_all(tr, d, cyc)
out = ll.copy()
for c in ("peak", "trough", "mkt_peak", "mkt_trough"):
    out[c] = out[c].dt.strftime("%Y-%m")
display(out[["peak", "trough", "mkt_peak", "mkt_trough", "lead_peak", "lead_trough",
             "dd_at_nber_peak", "max_dd", "censored", "tape"]])
s_all, s_bear = st.lead_summary(ll), st.lead_summary(ll, min_fall=0.10)
pd.DataFrame([s_all, s_bear], index=["all uncensored", "falls >= 10%"])[
    ["n", "trough_pos", "trough_neg", "trough_p", "trough_median", "peak_pos", "peak_p",
     "peak_median"]]
"""),
        md(r"""
> 💡 **In plain words.** Against a coin flip the lead looks overwhelming (sign-test p ≈ 0.001).
> The next cell shows why a coin flip is the wrong yardstick.

### 3.2 The honest null for the lead: shift the whole NBER calendar
"""),
        code(r"""
lr = st.lead_rotation_test(tr, cyc)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].hist(lr["perm_share"] * 100, bins=15, color=GREY)
axes[0].axvline(lr["observed_share"] * 100, color=RED, lw=3, label="real NBER calendar")
axes[0].set_xlabel("% of cycles with the market trough first"); axes[0].legend(fontsize=9)
axes[1].hist(lr["perm_mean_lead"], bins=40, color=GREY)
axes[1].axvline(lr["observed_mean_lead"], color=RED, lw=3)
axes[1].set_xlabel("mean trough lead, months")
fig.suptitle(f"Same estimator, NBER calendar shifted by every k months ({lr['n_shifts']} shifts)")
plt.tight_layout(); plt.show()
print(f"share first: {lr['observed_share']:.0%} vs {lr['null_share_mean']:.0%}  p = {lr['p_share']:.3f}")
print(f"mean lead:   {lr['observed_mean_lead']:.1f} vs {lr['null_mean_lead']:.1f} m  p = {lr['p_mean_lead']:.2f}")
"""),
        md(r"""
The estimator itself puts "the market first" {rot_share_null} of the time on arbitrary dates —
on an upward-drifting index the lowest point after a high tends to come early. Against that
null the *direction* still clears (p = {rot_p}); the *size* (mean {rot_mean_obs} m vs
{rot_mean_null} m on shifted calendars, p = {rot_mean_p}) does not. The synthetic null caught
this lean before the real tape was read with this test; the verdict's lead leg requires the
rotation p as well as the sign test.

> 💡 **In plain words.** The market reliably turns first; how *far* first is not a stable number.
"""),
        code(r"""
rows = []
for slack in (0, 3, 6):
    for post in (6, 12, 18):
        g = st.lead_rotation_test(tr, cyc, peak_slack=slack, post_months=post)
        rows.append({"peak_slack": slack, "post_months": post, "share": g["observed_share"],
                     "null_share": g["null_share_mean"], "p_share": g["p_share"]})
pd.DataFrame(rows)
"""),
        md(r"""
Window robustness: the share test clears p < 0.05 for {grid_sig} window choices.

### 3.3 What was already done on announcement day
"""),
        code(r"""
at_m = st.announcement_table(tr, anns)
at_d = st.announcement_table(d, anns, bars_per_month=21)
best = st.best_available(at_m, at_d)
best[["kind", "date", "lag_months", "already", "still_to_come", "share_done",
      "months_since_extreme", "fwd_3m", "fwd_12m", "fwd_24m", "tape"]]
"""),
        code(r"""
fig, ax = plt.subplots(figsize=(11, 5))
lvl = d / d.loc["2007-01-03"]
ax.plot(lvl.loc["2007":"2011"], color=BLUE, lw=1.4, label="S&P 500 price (rebased)")
for _, a in anns[(anns.date >= "2007-01-01") & (anns.date <= "2011-12-31")].iterrows():
    ax.axvline(a.date, color=RED if a.kind == "peak" else GREEN, lw=2,
               label=f"{a.kind} announced {a.date.date()}")
shade_recessions(ax, lo=pd.Timestamp("2007-01-01"), hi=pd.Timestamp("2011-12-31"))
ax.set_title("2007-2011: the NBER spoke after the fall and after the rebound")
ax.legend(fontsize=9); plt.show()
"""),
        md(r"""
Median drawdown already realised at peak announcements: {dd_already} ({no_fall} with no real
fall); median share of the cycle's fall already done {share_done}. At trough announcements:
median rebound already missed {missed} (minimum {missed_min}), the low a median {since_low}
months earlier.

### 3.4 The buy-signal test — exact rotation
"""),
        code(r"""
rows = []
for kinds, lbl in ((("peak", "trough"), "pooled"), (("peak",), "peak"), (("trough",), "trough")):
    for hm in (3, 6, 12, 24):
        for tape, lvl, bpm in (("monthly TR", xs, 1), ("daily price", d, 21)):
            r = st.rotation_test(lvl, st.event_positions(lvl.index, anns, kinds), hm * bpm)
            rows.append({"events": lbl, "h": hm, "tape": tape, "n": r["n_events"],
                         "obs": r["observed"], "random": r["null_mean"],
                         "p_up": r["p_one_sided"], "p_down": r["p_lower"]})
rt = pd.DataFrame(rows)
rt.pivot_table(index=["events", "h"], columns="tape", values=["obs", "random", "p_up"])
"""),
        code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, kinds, title in zip(axes, (("peak", "trough"), ("peak",), ("trough",)),
                            ("pooled (10)", "peaks (5)", "troughs (5)")):
    r = st.rotation_test(xs, st.event_positions(xs.index, anns, kinds), 12)
    ax.hist(r["perm"] * 100, bins=40, color=GREY)
    ax.axvline(r["observed"] * 100, color=RED, lw=3)
    ax.set_title(f"{title}: p = {r['p_one_sided']:.2f}")
    ax.set_xlabel("mean 12 m excess return, %")
plt.tight_layout(); plt.show()
"""),
        md(r"""
Primary result: {buy_obs} after the announcements vs {buy_null} after random dates,
**p = {buy_p}**; peaks alone p = {buy_p_peak}, troughs alone p = {buy_p_trough}; daily price tape
p = {buy_p_daily}. The trough subgroup sits in the *lower* tail (p↓ ≈ {trough_lower_p}) — one
tail of one subgroup among dozens of cells; recorded, not claimed.

> 💡 **In plain words.** The year after "it's official" looked like any other year — or a
> little worse.

### 3.5 Power on the synthetic world
"""),
        code(r"""
vol = float(m.mkt.std() * np.sqrt(12))
eff = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
pc10 = st.power_curve(effects=eff, n_sims=100, n_cycles=5, vol=vol)
pc6 = st.power_curve(effects=eff, n_sims=100, n_cycles=3, vol=vol, seed=2013)
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(pc10.index * 100, pc10.reject_rate * 100, "o-", color=BLUE, lw=2, label="10 events")
ax.plot(pc6.index * 100, pc6.reject_rate * 100, "s-", color=AMBER, lw=2, label="6 events")
ax.axhline(80, color=GREY, ls="--"); ax.axhline(5, color=GREY, ls=":")
ax.set_xlabel("planted abnormal 12-month log return (%)"); ax.set_ylabel("rejection rate, %")
ax.set_title("SYNTHETIC: power of the rotation test at the tape's volatility"); ax.legend()
plt.show()
print(f"MDE80: 10 events {st.minimum_detectable_effect(pc10):.0%}, "
      f"6 events {st.minimum_detectable_effect(pc6):.0%}  (100 worlds per point here; "
      f"results.md uses 200)")
"""),
        md(r"""
At the tape's ~18% volatility, 10 events reach 80% power only at a planted ≈ {mde80} abnormal
12-month return (6 events: ≈ {mde80_6}). The size row (0% effect) sits near 5%: the rotation test
is honest; it is the sample that is small.

### 3.6 Positive control (synthetic) — fires on planted, quiet on null
"""),
        code(r"""
rows = []
for s in (1.0, 0.0):
    w = data.synthetic_world(signal_strength=s, seed=1013)
    lvl = st.excess_index(w["returns"]); tri = (1 + w["returns"]["mkt"]).cumprod()
    r = st.rotation_test(lvl, st.event_positions(lvl.index, w["announcements"]), 12)
    lr_ = st.lead_rotation_test(tri, w["cycles"])
    rows.append({"signal_strength": s, "planted_lead": w["truth"]["lead_months"],
                 "lead_share_p": lr_["p_share"], "planted_premium": w["truth"]["announce_premium"],
                 "measured_excess": r["excess_vs_random"], "buy_p": r["p_one_sided"]})
pd.DataFrame(rows)
"""),
        md(r"""
## 4 · Could you trade it?

One execution lag (decided at bar *t*'s close, earns *t+1*); each switch trades 100% of NAV
at 10 bp one-way; Sharpe on excess-of-bills returns for every leg; active-return t is
Newey-West (12 lags monthly, 21 daily). Daily rows are price-only against 0% cash on both
sides.
"""),
        code(r"""
mm = m[m.index >= "1980-01-01"]
dr = d.pct_change().dropna(); rf0 = pd.Series(0.0, index=dr.index)
sig_m = {"bills while officially in recession": st.signal_avoid(mm.index, anns),
         "equity 12 m after each announcement": st.signal_buy_after(mm.index, anns, 12),
         "equity 24 m after each peak announcement": st.signal_buy_after(mm.index, anns, 24, ("peak",))}
sig_d = {"bills while officially in recession": st.signal_avoid(dr.index, anns),
         "equity 12 m after each announcement": st.signal_buy_after(dr.index, anns, 252),
         "equity 24 m after each peak announcement": st.signal_buy_after(dr.index, anns, 504, ("peak",))}
cm = st.compare_rules(mm.mkt, mm.rf, sig_m, 10, 12, hac_lags=12)
cd = st.compare_rules(dr, rf0, sig_d, 10, 252, hac_lags=21)
display(cm[["exposure", "switches", "sharpe_gross", "sharpe_net", "sharpe_gain_net",
            "active_hac_t", "max_dd_net"]])
cd[["exposure", "switches", "sharpe_gross", "sharpe_net", "sharpe_gain_net", "active_hac_t",
    "max_dd_net"]]
"""),
        code(r"""
key = "equity 12 m after each announcement"
rp = st.rule_rotation_test(mm.mkt, mm.rf, sig_m[key], 10, 12)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.hist(rp["perm"], bins=40, color=GREY, label="same rule, shifted dates (same exposure & costs)")
ax.axvline(rp["observed_sharpe"], color=RED, lw=3, label="official dates")
ax.axvline(cm.loc["buy-and-hold", "sharpe_net"], color=BLUE, lw=2, ls="--", label="buy-and-hold")
ax.set_xlabel("net excess Sharpe, 1980-2018"); ax.legend(fontsize=9); plt.show()
print(f"shifted-date p = {rp['p_one_sided']:.2f}")
st.cost_sweep(mm.mkt, mm.rf, sig_m[key], costs=(0, 5, 10, 25, 50))
"""),
        md(r"""
The buy-after rule's Sharpe ({rule_sharpe}) is below buy-and-hold ({bh_sharpe}) before costs,
so no break-even cost exists; and it is unremarkable against its own shifted-date versions
(p = {rule_perm_p}) — timing adds nothing, the lower exposure ({rule_exposure} in equity) just
forfeits premium. The sceptic's rule (bills while officially in recession) costs
{avoid_gain} of Sharpe monthly. Capacity is irrelevant: index switches a few times a decade.
**Mirage.**

> 💡 **In plain words.** Waiting for the referee loses whichever way you act on the whistle.

## 5 · Going further

- **More independent events.** Euro-area (CEPR) and other national dating committees would add
  announcements that are not the same twelve US dates; pooled, the MDE would roughly halve.
- **The trough curiosity** (below-random returns after "recovery" announcements) is worth a
  pre-registered test on that fresh sample.
- **Real-time recession signals** move earlier than the committee — 268-sahm-rule,
  626-unemployment-trend-timing and 881-jobless-claims-nowcast on this desk.
- **Price vs total-return dating.** Rerunning §3.1 on a price index (not shipped offline before
  1990) would move pre-1960 peaks earlier and troughs later.
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


if __name__ == "__main__":
    build_curious()
    build_quants()
