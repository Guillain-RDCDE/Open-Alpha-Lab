"""Notebook builder for Study 1026 — Gibson's Paradox.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both walk the same seven desk beats (METHODOLOGY.md) at two altitudes. The tapes are frozen
inside ``arch`` / ``statsmodels``, so every code cell below runs on the **real** data, offline;
the synthetic two-world control is labelled as such wherever it appears. The verdict badges and
the prose numbers come from ONE place — the ``REAL`` dict below, which mirrors
``docs/results.md`` (regenerate both together: run ``examples/verify.py`` first).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# The real run, mirrored from docs/results.md (as-of 2018-11-30). ONE place.
# --------------------------------------------------------------------------- #
REAL = {
    "as_of": "2018-11-30",
    "signal": "Mixed", "trad": "Fragile",
    "fp_us": "d269fafaad79", "fp_uq": "c617cf170e24", "fp_de": "1ac0ebf245e1",
    "corr_logp": "-0.00", "gi_corr_logp": "+0.92", "dis_corr_logp": "-0.96",
    "corr_gap": "+0.80", "corr_pi": "+0.75", "corr_ewma": "+0.91",
    "t_gap": "+3.56", "t_pi": "+0.38", "de_t_gap": "+0.07", "de_t_pi": "-1.00",
    "eg_p_gap": "0.19", "eg_p_pi": "0.28", "eg_p_ewma": "0.009",
    "naive_t_gap": "+36.8", "t_gap_vs_ewma": "+1.28",
    "cw_p_gap": "0.000", "cw_p_pi": "0.011", "cw_p_gap_rol": "0.82",
    "timer_diff": "+1.95%", "timer_t": "+2.63", "timer_sharpe": "0.54", "const_sharpe": "0.50",
    "timer_pos": "1.43", "timer_turn": "0.13", "timer_alpha": "+0.65%", "timer_alpha_t": "+1.01",
    "timer_gi": "-0.05%", "timer_dis": "+2.44%", "timer_vs_yield": "+3.22%",
    "breakeven": "1499",
}

SIG_COLOR = {"Real": "2ea44f", "Weak": "dab617", "Mixed": "dab617", "None": "c0392b"}
TRD_COLOR = {"Investable": "2ea44f", "Fragile": "dab617", "Mirage": "c0392b"}
BADGES = (
    f"![Signal: {REAL['signal']}](https://img.shields.io/badge/Signal-{REAL['signal']}-"
    f"{SIG_COLOR[REAL['signal']]}?style=flat-square)\n"
    f"![Tradability: {REAL['trad']}](https://img.shields.io/badge/Tradability-{REAL['trad']}-"
    f"{TRD_COLOR[REAL['trad']]}?style=flat-square)"
)

BOOT = """\
import sys, os, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.dpi"] = 80
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
pd.set_option("display.width", 160)
from gibson import data, strategy as st
us = data.load_us_monthly()
us["ewma"] = st.adaptive_expectation(us["pi_1m"], 36)
de = data.load_germany()
uq = data.load_us_quarterly()
print(f"US monthly {us.index[0]:%Y-%m} -> {us.index[-1]:%Y-%m} | fingerprint "
      f"{data.fingerprint(us[['aaa', 'baa', 'cpi', 'rf']])}")
print(f"Germany    {de.index[0]:%Y-%m} -> {de.index[-1]:%Y-%m} | fingerprint "
      f"{data.fingerprint(de[['R', 'Dp']])}")
"""


def md(text):
    return new_markdown_cell(text.strip("\n").rstrip())


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


R = REAL

VERDICT_SIGNAL = (
    f"**Signal: {R['signal']}** — *a US-only shadow of Fisher.* On the US tape the detrended "
    f"price level beats 12-month inflation in first differences (Newey-West t {R['t_gap']} vs "
    f"{R['t_pi']}) and out of sample (Clark-West p {R['cw_p_gap']}); in Germany 1972-98 it shows "
    f"nothing (t {R['de_t_gap']}). Yields are **not** cointegrated with the price gap (Engle-Granger "
    f"p {R['eg_p_gap']}) but **are** with a slow, adaptive inflation expectation (p "
    f"{R['eg_p_ewma']}) — Fisher's own 1930 explanation of the paradox. Detrend over a rolling "
    f"10-year window instead and the forecasting edge vanishes (p {R['cw_p_gap_rol']}).")
VERDICT_TRAD = (
    f"**Tradability: {R['trad']}.** A duration overlay on a 20-year AAA bond earns "
    f"{R['timer_diff']} a year net over constant duration (t {R['timer_t']}; net Sharpe "
    f"{R['timer_sharpe']} vs {R['const_sharpe']}), but it sits at {R['timer_pos']}× duration on "
    f"average, trades {R['timer_turn']}× a year, has an alpha of only {R['timer_alpha']} (t "
    f"{R['timer_alpha_t']}) after its extra duration, and made {R['timer_gi']} a year in 1965-81 "
    f"against {R['timer_dis']} in 1982-2018. One secular trade, not a repeatable rule.")


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(f"""
# Gibson's Paradox \U0001FA99
### Do bond yields follow the price level, or the inflation rate?

{BADGES}

For a hundred years before the First World War, British bond yields rose when **prices were
high** and fell when prices were low — not when prices were *rising*, as theory said they should.
Keynes called it Gibson's paradox. We asked whether today's version, "yields follow prices", holds
up on sixty years of fiat-money data, and whether it can tell you when to own long bonds.

> \U0001F4D3 **This is the plain-language layer.** Unit roots, cointegration, Clark-West and the
> full timing tables are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> ⚠️ **Not investment advice.** Every chart is drawn by the code beside it, on real tapes frozen
> inside Python packages (as-of {R['as_of']}). The headline numbers are mirrored from
> [`docs/results.md`](../docs/results.md).
"""),
        code(BOOT),
        md(f"""
## Beat 0 · The answer first (real tape)

| Question | Answer |
|---|---|
| Do yields and the price level move together? | On a chart, strikingly — but two trending lines always do. |
| Does it survive the tests that catch coincidences? | Half-way, in the US only. Not in Germany. |
| Is it really "the price level"? | Better read as slow-moving inflation expectations — Fisher's own answer. |
| Can it time long bonds? | It made money, but as one long bet on the 1982-2018 bond rally. |

{VERDICT_SIGNAL}

{VERDICT_TRAD}
"""),
        md("""
## 1 · The claim

> *"The price level tells you where yields go."* Under the classical gold standard, the
> correlation between British consols yields and wholesale prices was so tight that Keynes (1930)
> called it "one of the most completely established empirical facts within the whole field of
> quantitative economics". It was a paradox because Irving Fisher's theory says lenders demand
> compensation for **inflation** — the *change* in prices — not for the price *level*.

Barsky & Summers (1988) explained it as a feature of the gold standard. The modern investor's
version drops the history: commodity and consumer prices are high, so yields should be high too.
"""),
        md("""
## 2 · So what?

If yields really track the price level, a bond investor could read the direction of rates off the
CPI and own long duration only when prices are "low". On a 20-year bond, a 1-point move in yields
is roughly an 11% gain or loss, so even a modest edge would be worth a lot.

And it would matter for economics: it would mean bond markets do not price inflation the way the
textbook says.
"""),
        md("""
## 3 · How we'd know

Before running anything we wrote down the tests (they live in `gibson/strategy.py`):

1. **Do the levels share a trend for a reason?** Two series that both drift will correlate by
   accident. A real long-run link leaves a gap between them that keeps closing (*cointegration*).
2. **Do the *changes* line up?** When the price level jumps this month, do yields jump — even
   after allowing for the change in inflation?
3. **Does it forecast?** Using only what was known at the time (CPI is published a month late),
   does the price level improve a forecast of next year's yield change?

Two of three on the US tape plus at least one in Germany would make it **Real**. Nothing anywhere
would make it **None**. A timing rule that cannot beat holding the bond after costs is a
**Mirage**.

**What we cannot test:** these tapes have no price index before 1957, so the gold-standard
era where the paradox was born is out of reach. This is a test of the paradox under fiat money.
"""),
        md("""
## 4 · The teardown

### 4.1 The picture that started it — and why it fools you
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
ax = axes[0]
ax.plot(us.index, us["aaa"], color="#1f6feb", lw=1.8, label="AAA bond yield, %")
ax2 = ax.twinx()
ax2.plot(us.index, us["cpi"], color="#c0392b", lw=1.8, label="core CPI (price level)")
ax2.set_yscale("log"); ax2.grid(False)
ax.set_title("raw: prices only go up, yields go up then down", fontsize=10)
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
ax = axes[1]
ax.plot(us.index, us["aaa"], color="#1f6feb", lw=1.8, label="AAA yield, %")
ax2 = ax.twinx()
ax2.plot(us.index, us["gap_exp"], color="#c0392b", lw=1.8, label="price level vs its trend, %")
ax2.grid(False)
ax.set_title("detrended: the Gibson picture reappears", fontsize=10)
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()
for name, (a, b) in {"1965-81": data.GREAT_INFLATION, "1982-2018": data.DISINFLATION}.items():
    c = st.level_correlations(us, "aaa", ("logp", "gap_exp", "pi"), a, b)
    print(f"{name}: corr with raw price {c['logp']:+.2f} | with detrended price "
          f"{c['gap_exp']:+.2f} | with inflation {c['pi']:+.2f}")
"""),
        md("""
Under fiat money the raw price level only goes up, so it "correlates" positively with yields while
yields rise (1965-81) and negatively while they fall (1982-2018). Measured against its own trend,
the price level tracks yields beautifully. That is the modern paradox. Now the question is whether
the beautiful line means anything.

### 4.2 How easy it is to fool yourself
"""),
        code("""
rng = np.random.default_rng(1974)
ts = []
for _ in range(500):
    a = np.cumsum(rng.normal(size=600)); b = np.cumsum(rng.normal(size=600))
    ts.append(st.hac_ols(a, b, lags=0)["t"][1])
ts = np.abs(np.array(ts))
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(np.clip(ts, 0, 80), bins=40, color="#8b949e")
ax.axvline(2, color="#c0392b", lw=2, label="the usual 'significant' line, t = 2")
ax.set_xlabel("|t-statistic| from regressing one random walk on ANOTHER, unrelated one")
ax.legend(fontsize=9); plt.show()
print(f"{(ts > 2).mean():.0%} of 500 pairs of unrelated random walks look 'significant'.")
"""),
        md("""
These are pairs of **completely unrelated** random series, and most of them look "significantly"
related. Yields and the price level are both series of this drifting kind (the quant notebook runs
the formal unit-root tests). So the correlation in 4.1 is, by itself, worth nothing.

> \U0001F52C **For the quants.** This is Granger & Newbold (1974). The fix is to test for
> cointegration and to work in first differences — §3-§4 of the quant notebook.

### 4.3 Test 1 — do they share a trend for a reason?
"""),
        code("""
rows = []
for lab, x in (("price level vs trend", us["gap_exp"]), ("12-month inflation", us["pi"]),
               ("slow inflation expectation (3-yr memory)", us["ewma"])):
    r = st.engle_granger(us["aaa"], x)
    rows.append({"yield tied to...": lab, "naive t (the trap)": r["naive_t"],
                 "cointegration p-value": r["eg_p"]})
print(pd.DataFrame(rows).to_string(index=False))
"""),
        md(f"""
A small p-value means the yield and the variable really are tied together in the long run. The
price gap is **not** (p ≈ {R['eg_p_gap']}), despite a naive t-statistic of {R['naive_t_gap']}. But a
*slow-moving average of past inflation* — what investors might reasonably expect inflation to be
if they learn slowly — **is** (p ≈ {R['eg_p_ewma']}).

That is exactly how Irving Fisher answered Gibson in 1930: people's inflation expectations adjust
over years, so yields track a long, smoothed memory of inflation — which, drawn on a chart, looks
just like the price level.

### 4.4 Test 2 — do the changes line up?
"""),
        code("""
d = us[["aaa", "gap_exp", "pi"]].diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, col, lab in ((axes[0], "gap_exp", "monthly change in price level vs trend"),
                     (axes[1], "pi", "monthly change in 12-month inflation")):
    ax.scatter(d[col], d["aaa"], s=6, alpha=0.4, color="#1f6feb")
    b = np.polyfit(d[col], d["aaa"], 1)
    xs = np.linspace(d[col].quantile(0.01), d[col].quantile(0.99), 50)
    ax.plot(xs, np.polyval(b, xs), color="#c0392b", lw=2)
    ax.set_xlabel(lab, fontsize=9)
axes[0].set_ylabel("monthly change in AAA yield, pp")
plt.tight_layout(); plt.show()
hr = st.diff_horse_race(us, "aaa", "gap_exp", "pi")
print(f"US, both in the race:   price-level t {hr['t_gap']:+.2f} | inflation t {hr['t_pi']:+.2f}")
hd = st.diff_horse_race(de, "R", "gap_exp", "pi", lags=4)
print(f"Germany, both in race:  price-level t {hd['t_gap']:+.2f} | inflation t {hd['t_pi']:+.2f}")
"""),
        md("""
In the US, months with a big price jump are months when yields rise, and that beats the change in
the inflation *rate* as an explanation. In Germany neither variable explains anything. The US
result is a genuine fact about the data — but notice that "this month's price jump" is just this
month's inflation, which is also what moves a slow inflation expectation. The two stories make
the same prediction month to month.

### 4.5 Test 3 — could you have used it in real time?
"""),
        code("""
lag = st.add_publication_lag(us, cols=("gap_exp", "gap_rol", "pi", "ewma"))
tbl, fc = st.oos_table(lag, "aaa", 12, 120, variables=("gap_exp", "gap_rol", "pi", "ewma"))
names = {"gap_exp": "price vs long trend", "gap_rol": "price vs 10-yr trend",
         "pi": "12-month inflation", "ewma": "slow inflation expectation"}
out = tbl[["oos_r2_vs_yield_only", "cw_p"]].rename(index=names)
out.columns = ["forecast improvement (R² vs yield alone)", "p-value"]
print(out.to_string())
"""),
        md(f"""
Measured against its full-history trend, the price level does help forecast next year's yield
change in real time. But measured against a 10-year trend — the same idea, a different but equally
reasonable ruler — it does not (p ≈ {R['cw_p_gap_rol']}). An edge that depends on how you draw
the trend line is a fragile edge.
"""),
        md("""
## 5 · The verdict
"""),
        md(f"""
{VERDICT_SIGNAL}

It is **Mixed**, not Real, because the result splits by country (US yes, Germany no) and because
the US half reads better as Fisher with a long memory than as a law about price levels.
"""),
        md("""
## 6 · Could you trade it?
"""),
        code("""
bond = st.bond_returns(us["aaa"], us["rf"])
pos = st.timing_positions(fc["gap_exp"])
bt = st.timing_backtest(bond, pos, cost_bps=5.0)
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
ax = axes[0]
ax.plot(bt.index, (1 + bt["net"]).cumprod(), color="#c0392b", lw=2,
        label="price-level timer, net of 5 bp (excess of cash)")
ax.plot(bt.index, (1 + bt["const"]).cumprod(), color="#1f6feb", lw=2,
        label="just hold the bond (excess of cash)")
ax.set_yscale("log"); ax.legend(fontsize=9)
ax.set_title("20-year AAA bond, growth of $1 above cash", fontsize=10)
axes[1].step(bt.index, bt["pos"], where="post", color="#8b949e")
axes[1].set_ylabel("duration held (×)")
plt.tight_layout(); plt.show()
s = st.timing_summary(bt, splits={"1965-81": data.GREAT_INFLATION,
                                  "1982-2018": data.DISINFLATION})
print(f"extra return vs holding: {s['diff_net_ann']:+.2%} a year (t {s['diff_net_t']:+.2f})")
for k, v in s["subs"].items():
    print(f"   {k}: {v['diff_net_ann']:+.2%} a year")
print(f"average duration held {s['mean_pos']:.2f}x, trades {s['turnover_ann']:.2f}x a year")
"""),
        md(f"""
The timer did beat simply holding the bond. Look at the bottom panel, though: it went long
duration in the 1980s and essentially **stayed there** while yields fell for thirty-five years. It
made nothing in the 1965-81 inflation, the one period where yields went the "wrong" way. Strip out
the extra duration it was carrying and the skill left over is {R['timer_alpha']} a year with a
t-statistic of {R['timer_alpha_t']} — indistinguishable from luck.

{VERDICT_TRAD}

> \U0001F52C **For the quants.** Excess-vs-excess Sharpe, one execution lag plus a one-month CPI
> lag, 5 bp one-way per unit traded, and a duration-plus-convexity approximation for the bond.
> The cost sweep and the yield-only control timer are in §6 of the quant notebook.
"""),
        md("""
## 7 · Going further \U0001F6AA

- **Test the gold standard.** Add a pre-1914 price index (e.g. UK wholesale prices) and the
  Moody's tape back to 1919 — the paradox should come back where Barsky & Summers say it lives.
- **Model expectations directly.** Survey expectations (Michigan, SPF) instead of a smoothed
  average would settle whether "price level" or "slow expectations" is the better story.
- **More countries.** One German sample of 27 years is thin; a panel of fiat-era bond markets
  would give the cross-country leg real power.
- **Fork it:** everything runs offline from frozen tapes — `examples/verify.py` regenerates every
  number in under three minutes.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(f"""
# Gibson's Paradox — a quantitative teardown \U0001F52C
### Unit roots · Engle-Granger · a first-difference horse race · Clark-West · a duration overlay

{BADGES}

The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim
(Keynes 1930): long yields co-move with the **price level**, not with **inflation**. The tests
that carry the verdict are §3 (cointegration), §4 (differences, Newey-West), §5 (the
adaptive-expectations confound) and §6 (out of sample + the overlay). §7 is the synthetic
two-world control — a machinery proof, never market evidence.

> ⚠️ **Not investment advice.** Real tapes frozen inside `arch` / `statsmodels` (SHA-256 pinned
> through `quantlab.bundled`), as-of {R['as_of']}; fingerprints US `{R['fp_us']}`, US quarterly
> `{R['fp_uq']}`, Germany `{R['fp_de']}`. Numbers in prose mirror
> [`docs/results.md`](../docs/results.md).
>
> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT),
        md(f"""
## Verdict, up front (real tape)

| Axis | Stamp | Why |
|---|---|---|
| **Signal** | {R['signal']} | {VERDICT_SIGNAL} |
| **Tradability** | {R['trad']} | {VERDICT_TRAD} |

> \U0001F4A1 **In plain words.** The US data do contain a Gibson-looking pattern, but it is best
> explained by investors' slow-moving inflation expectations, it is absent in Germany, and the
> trading rule it implies was one long bet that happened to be right.
"""),
        md("""
## 1 · Hypotheses and pre-registered rules

Variables: *Y* = Moody's AAA yield (%), *p* = 100·log core CPI, gap = *p* minus a past-only
linear trend (expanding from 1957; rolling 10-year as robustness), π = 12-month log inflation.

For each side — **Gibson** (gap) and **Fisher** (π) — three legs:

- **L1** Engle-Granger p < 0.05 for *Y* on the variable.
- **L2** Newey-West t ≥ 2 on the variable's slope in the *joint* regression ΔY = a + b·Δgap + c·Δπ.
- **L3** Clark-West one-sided p < 0.05 for adding the variable (lagged one month for CPI
  publication) to a yield-only expanding-window forecast of Y[t+12] − Y[t].

**Signal** (`signal_stamp`): *Real* = US has L2 and (L1 or L3) **and** Germany passes ≥ 1 leg;
*Mixed* = one tape strong, the other nothing; *Weak* = any leg anywhere; *None* = no leg.
**Tradability** (`trad_stamp`): *Investable* = overlay net − constant HAC t ≥ 2, positive in both
regimes, beats the yield-only timer, with a Real/Mixed signal; *Fragile* = positive net edge and
higher net Sharpe; else *Mirage*.
"""),
        md("""
## 2 · Levels and unit roots
"""),
        code("""
cols = ("logp", "gap_exp", "gap_rol", "pi", "ewma")
rows = []
for name, (a, b) in {"1957-2018": (None, None), "1965-81": data.GREAT_INFLATION,
                     "1982-2018": data.DISINFLATION}.items():
    rows.append({"tape": "US", "period": name, **st.level_correlations(us, "aaa", cols, a, b)})
rows.append({"tape": "US quarterly (headline)", "period": "1959-2009",
             **st.level_correlations(uq, "aaa", cols[:4])})
rows.append({"tape": "Germany", "period": "1972-98", **st.level_correlations(de, "R", cols[:4])})
print("Correlation of the long yield with... (descriptive only)")
print(pd.DataFrame(rows).set_index(["tape", "period"]).round(2).to_string())
ur = st.unit_root_table({"AAA": us["aaa"], "ΔAAA": us["aaa"].diff(), "log CPI": us["logp"],
                         "1m inflation": us["pi_1m"], "12m inflation": us["pi"],
                         "gap (expanding)": us["gap_exp"], "gap (rolling)": us["gap_rol"],
                         "adaptive exp. 36m": us["ewma"], "DE R": de["R"], "DE 4q infl": de["pi"]})
print(); print(ur[["n", "adf_p", "kpss_p", "i1_like"]].round(3).to_string())
"""),
        md("""
> \U0001F4A1 **In plain words.** Every level that matters drifts like a random walk — ADF cannot
> reject a unit root, KPSS rejects stationarity. Level correlations between such series are the
> textbook spurious regression, so the table above is a description of the claim, not evidence.
> Note also that if inflation is I(1) the log price level is I(2): a raw price level cannot be
> cointegrated with an I(1) yield at all, which is why only the detrended gap is a fair test.
"""),
        md("""
## 3 · Cointegration — Engle-Granger
"""),
        code("""
rows = []
for lab, x in (("raw log price", us["logp"]), ("gap (expanding)", us["gap_exp"]),
               ("gap (rolling 10y)", us["gap_rol"]), ("12m inflation", us["pi"]),
               ("adaptive exp. 36m", us["ewma"])):
    r = st.engle_granger(us["aaa"], x)
    rows.append({"tape": "US", "Y on": lab, "slope": r["slope"], "naive t": r["naive_t"],
                 "R2": r["r2"], "EG p": r["eg_p"]})
for lab, x in (("gap (expanding)", de["gap_exp"]), ("4q inflation", de["pi"])):
    r = st.engle_granger(de["R"], x)
    rows.append({"tape": "Germany", "Y on": lab, "slope": r["slope"], "naive t": r["naive_t"],
                 "R2": r["r2"], "EG p": r["eg_p"]})
eg = pd.DataFrame(rows)
print(eg.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.barh(eg["tape"] + " · " + eg["Y on"], eg["EG p"],
        color=["#2ea44f" if p < 0.05 else "#8b949e" for p in eg["EG p"]])
ax.axvline(0.05, color="#c0392b", ls="--", lw=1.5, label="5%")
ax.set_xlabel("Engle-Granger p (H0: no cointegration)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** The naive t-statistics are enormous; the cointegration tests are
> not impressed — except for the slow inflation expectation. The yield has a genuine long-run
> anchor, and it is Fisher's, with a long memory.
"""),
        md("""
## 4 · First differences — the horse race
"""),
        code("""
rows = []
for name, (a, b) in {"1957-2018": (None, None), "1965-81": data.GREAT_INFLATION,
                     "1982-2018": data.DISINFLATION}.items():
    rows.append({"tape": "US AAA", "period": name,
                 **st.diff_horse_race(us, "aaa", "gap_exp", "pi", 12, a, b)})
rows.append({"tape": "US BAA", "period": "1957-2018",
             **st.diff_horse_race(us, "baa", "gap_exp", "pi", 12)})
rows.append({"tape": "US quarterly (headline)", "period": "1959-2009",
             **st.diff_horse_race(uq, "aaa", "gap_exp", "pi", 4)})
rows.append({"tape": "Germany", "period": "1972-98",
             **st.diff_horse_race(de, "R", "gap_exp", "pi", 4)})
hr = pd.DataFrame(rows).set_index(["tape", "period"])
print(hr[["n", "b_gap", "t_gap", "b_pi", "t_pi", "t_gap_alone", "t_pi_alone"]].round(3).to_string())
rd = st.regime_difference(us, "aaa", data.GREAT_INFLATION, data.DISINFLATION)
print(f"\\ngap slope, 1982-2018 minus 1965-81: {rd['diff_gap']:+.3f} (HAC t {rd['t_diff_gap']:+.2f})")
"""),
        md("""
> \U0001F4A1 **In plain words.** In changes, the US yield responds to this month's price move
> (which *is* this month's inflation) more than to the change in the 12-month inflation rate — on
> core and headline CPI, for AAA and BAA. Germany shows nothing. The regime split is not
> significant, so we do not claim the effect "faded".
"""),
        md("""
## 5 · The confound — Fisher with a long memory
"""),
        code("""
rows = []
for hl in (12, 36, 60, 120):
    e = st.adaptive_expectation(us["pi_1m"], hl)
    dd = us.assign(ew=e)
    r = st.diff_horse_race(dd, "aaa", "gap_exp", "ew", 12)
    ch = dd[["gap_exp", "ew"]].diff().dropna()
    rows.append({"half-life (m)": hl, "corr level": us["aaa"].corr(e),
                 "EG p": st.engle_granger(us["aaa"], e)["eg_p"],
                 "corr(Δgap, Δexp)": ch["gap_exp"].corr(ch["ew"]),
                 "t gap (joint)": r["t_gap"], "t exp (joint)": r["t_pi"]})
lm = pd.DataFrame(rows).set_index("half-life (m)")
print(lm.round(3).to_string())
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(us.index, us["aaa"], color="#1f6feb", lw=2, label="AAA yield")
z = us["ewma"]; ax.plot(us.index, z, color="#2ea44f", lw=2, label="adaptive inflation exp. (36m)")
g = us["gap_exp"]; sc = us["aaa"].std() / g.std()
ax.plot(us.index, (g - g.mean()) * sc + us["aaa"].mean(), color="#c0392b", lw=1.4, alpha=0.8,
        label="price gap (rescaled)")
ax.legend(fontsize=9); ax.set_ylabel("%"); plt.tight_layout(); plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** The whole half-life grid is shown, not a chosen point. In changes,
> the price gap and a slow inflation expectation are almost the same series (correlation ≥ 0.9),
> so no regression can tell them apart; in levels, the expectation fits better *and* is
> cointegrated with the yield. Fisher (1930) and Sargent (1973) read Gibson exactly this way.
"""),
        md("""
## 6 · Out of sample, and the overlay

### 6.1 Expanding-window forecasts with Clark-West
"""),
        code("""
lag = st.add_publication_lag(us, cols=("gap_exp", "gap_rol", "pi", "ewma"))
tabs, fcs = [], {}
for h in (3, 12):
    t, fc = st.oos_table(lag, "aaa", h, 120, variables=("gap_exp", "gap_rol", "pi", "ewma"))
    tabs.append(t.assign(tape="US")); fcs[h] = fc
lde = st.add_publication_lag(de, cols=("gap_exp", "pi"))
for h in (1, 4):
    t, _ = st.oos_table(lde, "R", h, 40, variables=("gap_exp", "pi"))
    tabs.append(t.assign(tape="Germany"))
oos = pd.concat(tabs).reset_index()
print(oos[["tape", "h", "variable", "n", "oos_r2_vs_yield_only", "cw_t", "cw_p",
           "yield_only_oos_r2_vs_mean"]].round(3).to_string(index=False))
"""),
        md("""
> \U0001F4A1 **In plain words.** The full-history gap forecasts in real time; the 10-year-trend gap
> does not; the slow expectation does as well as the gap. Germany's ~40-60 forecasts show nothing.
> Note that the yield-only model loses to the prevailing mean (negative last column): the
> benchmark is beatable, which flatters any added variable a little.
"""),
        md("""
### 6.2 The duration overlay

Bond return (stated): r[t+1] ≈ Y[t]/12 − D[t]·ΔY + ½·C[t]·ΔY², D and C of a 20-year semi-annual
par bond at the AAA yield. Weight 1.5× when the 12-month forecast is below the prevailing mean
change, else 0.5×; set at *t*, earns *t+1*; CPI lagged a month; 5 bp one-way × NAV traded.
Benchmark 1× constant duration; all excess of the T-bill.
"""),
        code("""
bond = st.bond_returns(us["aaa"], us["rf"])
fc12 = fcs[12]
splits = {"1965-81": data.GREAT_INFLATION, "1982-2018": data.DISINFLATION}
bts = {v: st.timing_backtest(bond, st.timing_positions(fc12[v]), 5.0)
       for v in ("gap_exp", "gap_rol", "pi", "ewma")}
bts["yield_only"] = st.timing_backtest(bond, st.timing_positions(fc12["gap_exp"], "f_base"), 5.0)
rows = []
for v, bt in bts.items():
    s = st.timing_summary(bt, splits=splits)
    rows.append({"timer": v, "net Sharpe": s["net"]["sharpe"], "const Sharpe": s["const"]["sharpe"],
                 "net-const /yr": s["diff_net_ann"], "HAC t": s["diff_net_t"],
                 "mean pos": s["mean_pos"], "turnover/yr": s["turnover_ann"],
                 "alpha /yr": s["alpha_ann"], "alpha t": s["alpha_t"],
                 **{f"{k} net-const": x["diff_net_ann"] for k, x in s["subs"].items()}})
print(pd.DataFrame(rows).set_index("timer").round(3).to_string())
g, y = bts["gap_exp"], bts["yield_only"]
c = g.index.intersection(y.index)
m = st.hac_mean(g.loc[c, "net"] - y.loc[c, "net"], 12)
print(f"\\ngap timer minus yield-only timer: {m['mean']*12:+.2%} /yr (t {m['t']:+.2f})")
sweep = []
for cb in (0, 2, 5, 10, 25, 50):
    s = st.timing_summary(st.timing_backtest(bond, st.timing_positions(fc12["gap_exp"]), cb))
    sweep.append({"cost bp": cb, "net-const /yr": s["diff_net_ann"], "t": s["diff_net_t"]})
print(pd.DataFrame(sweep).round(4).to_string(index=False))
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axes[0]
for v, col in (("gap_exp", "#c0392b"), ("ewma", "#2ea44f"), ("yield_only", "#8b949e")):
    bt = bts[v]
    ax.plot(bt.index, (bt["net"] - bt["const"]).cumsum() * 100, color=col, lw=2, label=v)
ax.axhline(0, color="k", lw=1); ax.legend(fontsize=9)
ax.set_title("cumulative net − constant duration, pp", fontsize=10)
ax = axes[1]
g = bts["gap_exp"]
roll = (g["net"] - g["const"]).rolling(60).mean() * 12 * 100
ax.plot(roll.index, roll, color="#c0392b", lw=2)
ax.axhline(0, color="k", lw=1)
ax.set_title("gap timer: rolling 5-year edge, pp/yr", fontsize=10)
plt.tight_layout(); plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** Costs are irrelevant — the rule barely trades. What matters is
> that it held ~1.4× duration through a 35-year bond rally and earned nothing in 1965-81. After
> regressing on constant duration the alpha is not significant. Fragile: real money on paper,
> one bet in substance.
"""),
        md("""
## 7 · Synthetic control — Fisher world vs Gibson world (machinery proof)

Same simulated price path; the yield follows Fisher (s = 0), Gibson (s = 1) or a blend. The
detectors must fire in their own world and stay quiet in the other. **Not market evidence.**
"""),
        code("""
rows = []
for s_ in (0.0, 0.25, 0.5, 0.75, 1.0):
    w, _ = data.synthetic_world(n_months=600, signal_strength=s_, seed=1026)
    hr_ = st.diff_horse_race(w, "aaa", "gap_exp", "pi", 12)
    rows.append({"s": s_, "t gap": hr_["t_gap"], "t infl": hr_["t_pi"],
                 "EG p gap": st.engle_granger(w["aaa"], w["gap_exp"])["eg_p"],
                 "EG p infl": st.engle_granger(w["aaa"], w["pi"])["eg_p"]})
syn = pd.DataFrame(rows).set_index("s")
print(syn.round(3).to_string())
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(syn.index, syn["t gap"], "o-", color="#c0392b", lw=2, label="t on price gap (Gibson)")
ax.plot(syn.index, syn["t infl"], "s-", color="#1f6feb", lw=2, label="t on inflation (Fisher)")
ax.axhline(2, color="k", ls="--", lw=1)
ax.set_xlabel("signal_strength (0 = Fisher world, 1 = Gibson world)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** The harness can tell the two worlds apart: the Gibson t rises with
> the planted strength and the Fisher t falls. So the real-tape result — a Gibson-looking
> difference signal *and* Fisher-type cointegration — is a finding about the data, not a blind spot
> of the tests.
"""),
        md("""
## 8 · Caveats

- **No gold standard on these tapes** (no price index before 1957). The paradox's home era is
  untested; this is the fiat-era version only.
- **Moody's AAA** is a seasoned corporate index with drifting maturity and a credit spread; the
  par-bond return is an approximation, and a real overlay would use Treasury futures (basis risk).
- **Final-vintage macro**; core CPI lagged one month for publication.
- **Specification sensitivity**: expanding vs rolling detrend changes the forecasting answer.
- **Germany is short** — low power, but no hint of the effect either.

## 9 · Going further \U0001F6AA

- A pre-1914 price index would let the same harness test the gold standard directly.
- Survey expectations (SPF, Michigan) in place of the exponential average.
- A cross-country fiat-era panel for a properly powered second leg.
- Fork `gibson/strategy.py`: every leg is a function, every threshold is a named rule.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


def _meta():
    return {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}


def _write(nb, name):
    path = os.path.join(HERE, name)
    nbf.write(nb, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    build_curious()
    build_quants()
