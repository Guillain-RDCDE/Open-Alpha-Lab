"""Notebook builder for Study 1027 — Oil Is Equity in a Mirror.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    cd notebooks
    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks walk the same seven desk beats (METHODOLOGY.md) at two altitudes: 01 is the
plain-language layer, 02 carries the inference. Every chart recomputes from the frozen real
tapes (offline). The verdict badges and the quoted headline numbers live in ``REAL`` below,
copied from ``docs/results.md`` (the output of ``examples/verify.py``) — never chosen by hand.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# The real run, mirrored from docs/results.md (examples/verify.py). One dict, one place.
REAL = {
    "signal": "None", "trad": "Mirage",
    "wti_gjr_inv_t": -2.62, "wti_egarch_inv_t": -4.40, "wti_mf_t": -1.28,
    "sp_gjr_inv_t": -8.10, "wti_gjr_ratio": 1.48,
    "era_pre_t": -0.15, "era_post_t": -4.81, "era_z": 3.38,
    "roll_share_inverse": "9%", "roll_last_inverse": "1997",
    "ov_wti_gain": -0.08, "ov_sp_gain": +0.02, "ov_wti_react": +0.04, "ov_sp_react": +0.31,
}

BADGE = {"Real": "2ea44f", "Weak": "dab617", "Mixed": "dab617", "None": "c0392b",
         "Investable": "2ea44f", "Fragile": "dab617", "Mirage": "c0392b"}


def badges():
    s, t = REAL["signal"], REAL["trad"]
    return (f"![Signal: {s}](https://img.shields.io/badge/Signal-{s}-{BADGE[s]}?style=flat-square)\n"
            f"![Tradability: {t}](https://img.shields.io/badge/Tradability-{t}-{BADGE[t]}"
            f"?style=flat-square)\n"
            f"![Mirror?: Busted](https://img.shields.io/badge/Mirror%3F-Busted-8b949e"
            f"?style=flat-square)")


BOOT = """\
import sys, os, warnings
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
warnings.filterwarnings("ignore")
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 90
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
OIL, EQ, NULL = "#eb6834", "#2a78d6", "#8b949e"     # oil = orange, equity = blue
from mirrorlev import data, strategy as st
"""


def md(text):
    return new_markdown_cell(text.strip("\n").rstrip())


def code(text):
    return new_code_cell(text.strip("\n").rstrip())


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    R = REAL
    cells = [
        md(f"""
# Oil Is Equity in a Mirror \U0001FA9E
### In stocks, the storm comes after the fall. In oil, they say, it comes after the rally. Does it?

{badges()}

> **In one sentence:** on 1986-2018 WTI spot, oil is not equity in a mirror but a fainter
> copy — its volatility also rises more after falls (GJR t = {R['wti_gjr_inv_t']:+.1f}), a tilt
> that appeared after 2008 — and a volatility-target overlay does not pay on the oil tape
> (net Sharpe gain {R['ov_wti_gain']:+.2f}), which is a spot price you could not hold anyway.

> \U0001F4D3 **This is the plain-language layer.** The GARCH fits, Newey-West errors, the
> rolling windows and the bootstrap intervals are in
> **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. The fingerprinted run is
> [docs/results.md](../docs/results.md).
>
> ⚠️ **Not investment advice.** Every chart below is computed by the code beside it, from
> real tapes frozen inside the `arch` Python package.
"""),
        code(BOOT),
        md("""
## The answer first

| Question | Answer |
|---|---|
| Do stocks get stormier after falls? | Yes, overwhelmingly. |
| Does oil get stormier after *rallies*? | No — not on this tape, in any pre-chosen era. |
| Then what does oil do? | A weaker version of what stocks do, and only since 2008. |
| Does a "turn it down when it gets loud" rule help on oil? | Not here, and not after costs. |
"""),
        md("""
## 1 · The claim

Every options trader knows the stock-market shape: prices fall, fear spikes, volatility
jumps. Rallies are calm. Fischer Black named it the **leverage effect** in 1976.

The commodity folk wisdom says oil is the *mirror image*. What really moves oil is
**supply**: a war, a pipeline cut, an OPEC surprise. Those shocks send the price *up* — and
that is exactly when everyone panics. So in oil, the story goes, it's the **rallies** that
bring the storm, and the slow grinding declines that are calm. Researchers have reported this
"inverse" asymmetry for gold and, in some studies, for energy futures.
"""),
        md("""
## 2 · So what?

If it were true, the most popular risk tool in the business would behave backwards on oil.
A **vol-target overlay** — hold more when markets are quiet, less when they are loud — is
everywhere: risk-parity funds, CTAs, target-volatility products. On stocks it ends up selling
*after falls*, because that's when volatility appears. On oil, the mirror says, it would sell
*after rallies* — trimming the winners, not dodging the crash. That changes what the tool is
for.

It would also say something deep: that the "leverage effect" is not about leverage at all, but
about **which direction the important news moves the price**.
"""),
        md("""
## 3 · How we'd know

Before looking, we wrote down what would count:

- **The mirror is real** if oil's volatility rises more after rallies than after equal-sized
  falls — measured two independent ways (a standard volatility model, and a simple
  "how loud is the next month?" regression), each clearing the desk's bar (a robust *t* of 2).
- **It's a regime, not a law,** if it holds clearly in one era and clearly reverses in another.
  We fixed the split in advance: **1986-2007** (the supply-shock era: the 1990 Gulf War, OPEC
  cuts) vs **2008-2018** (the demand-driven crashes of 2008 and 2014-16).
- **It's a mirage** if neither shows up — or if the sign is the *equity* one.

The data: daily **WTI spot** 1986-2018 (a spot price — more on that in beat 6), the daily
**S&P 500** 1999-2018 as the equity mirror, and monthly Brent as a cross-check.
"""),
        md("""
## 4 · The teardown

### The tapes
"""),
        code("""
wti, sp = data.load_wti(), data.load_sp500()
rw, rs = data.log_returns(wti), data.log_returns(sp)
fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
axes[0].plot(wti.index, wti, color=OIL, lw=1.0, label="WTI spot, $/bbl")
axes[0].plot(sp.index, sp / 20, color=EQ, lw=1.0, label="S&P 500 price index ÷ 20")
axes[0].set_ylabel("level"); axes[0].legend(loc="upper left", fontsize=9, frameon=False)
axes[1].plot(st.realised_vol(rw, 63).index, st.realised_vol(rw, 63) * 100, color=OIL, lw=1.0,
             label="WTI, 3-month realised vol")
axes[1].plot(st.realised_vol(rs, 63).index, st.realised_vol(rs, 63) * 100, color=EQ, lw=1.0,
             label="S&P 500, 3-month realised vol")
axes[1].set_ylabel("% a year"); axes[1].legend(loc="upper left", fontsize=9, frameon=False)
axes[0].set_title("Oil is about twice as volatile as stocks - and its storms come at both ends")
plt.tight_layout(); plt.show()
print(f"WTI: {len(rw):,} days {rw.index[0].date()} -> {rw.index[-1].date()}  |  "
      f"S&P: {len(rs):,} days {rs.index[0].date()} -> {rs.index[-1].date()}")
"""),
        md("""
Look at the oil vol spikes: 1986 (a price *collapse* when Saudi Arabia opened the taps),
1990-91 (a price *spike* when Iraq invaded Kuwait), 2008 and 2014-16 (two *collapses*).
Oil's storms really do come from both directions. The question is which direction wins on
average.

### What happens the month after a big day?

For every day, we look at how loud the *next* month is, and compare days that went up with
days that went down **by the same amount** (big down days are more common, so comparing
like with like matters).
"""),
        code("""
def after_moves(r, h=21, q=5):
    df = pd.DataFrame({"r": r, "fwd": st.forward_vol(r, h)}).dropna()
    df["size"] = pd.qcut(df["r"].abs(), q, labels=False)
    up = df[df.r > 0].groupby("size")["fwd"].mean()
    dn = df[df.r < 0].groupby("size")["fwd"].mean()
    return (up / dn)

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(5)
ax.bar(x - 0.2, after_moves(rw).values, 0.38, color=OIL, label="WTI 1986-2018")
ax.bar(x + 0.2, after_moves(rs).values, 0.38, color=EQ, label="S&P 500 1999-2018")
ax.axhline(1.0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels(["smallest", "small", "middle", "big", "biggest"])
ax.set_xlabel("size of today's move (quintile)")
ax.set_ylabel("next-month vol after UP ÷ after DOWN")
ax.set_ylim(0.6, 1.2); ax.legend(frameon=False, fontsize=9)
ax.set_title("Above 1 would be the mirror. Neither asset gets there.")
plt.show()
"""),
        md("""
Above the line would mean "rallies are followed by more noise than falls" — the mirror.
The S&P sits clearly below the line, as the textbook says. **Oil also sits below it** — just
closer to 1. After an up day, oil is a little *calmer* than after a down day of the same
size. That's the equity pattern, faintly — not its reflection.

> \U0001F52C **For the quants:** the formal version is a regression of forward 21-day realised
> vol on today's up-move and down-move separately, controlling for current vol, with
> Newey-West errors. Full-sample WTI: *t*(b_up − b_dn) = **"""
           + f"{R['wti_mf_t']:+.2f}" + """**; GJR-GARCH inverse *t* = **"""
           + f"{R['wti_gjr_inv_t']:+.2f}" + """** (negative = equity sign).

### Was the mirror ever there? Two eras
"""),
        code("""
eras = {"1986-2007": ("1986-01-01", "2007-12-31"), "2008-2018": ("2008-01-01", "2018-12-31")}
tbl = st.era_table(rw, eras)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(tbl.index, tbl["gjr_inv_t"], color=[OIL, OIL], width=0.5)
ax.axhline(2, color="k", ls="--", lw=1); ax.axhline(-2, color="k", ls="--", lw=1)
ax.axhline(0, color="k", lw=1)
ax.text(1.3, 2.15, "mirror clears the bar", fontsize=9, ha="right")
ax.text(1.3, -2.45, "equity sign clears the bar", fontsize=9, ha="right")
ax.set_ylabel("asymmetry score, t-stat\\n(+ = storms after rallies)")
ax.set_title("WTI: symmetric before 2008, equity-like after")
plt.show()
print(tbl[["n", "gjr_inv_t", "mf_t"]].round(2).to_string())
"""),
        md("""
The supply-shock era (1986-2007) is **symmetric** on average — rallies and falls stir up
about the same amount of noise. Then, from 2008, oil starts behaving like a stock: falls
are what make it loud. Something did change — but **away** from the mirror, not toward it.

There *are* stretches where oil looked like the mirror: three-year windows around the 1990-91
Gulf War and the tight-inventory run-up of 1996. That's the grain of truth in the story — a
real supply shock really can make rallies the scary direction, for a while. None of those
windows ends after """ + R["roll_last_inverse"] + """.

### And the vol-target overlay?
"""),
        code("""
rf = data.load_rf_daily(sp.index)
ow = st.vol_target_overlay(data.simple_returns(wti), None, cost_bps=10, n_boot=200)
os_ = st.vol_target_overlay(data.simple_returns(sp), rf, cost_bps=5, n_boot=200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, o, c, name in ((axes[0], ow, OIL, "WTI spot (proxy)"), (axes[1], os_, EQ, "S&P 500")):
    s = o["series"]
    ax.plot(np.exp(np.log1p(s["bh"]).cumsum()), color=NULL, lw=1.0, label="buy & hold")
    ax.plot(np.exp(np.log1p(s["net"]).cumsum()), color=c, lw=1.2, label="vol-target, net")
    ax.set_yscale("log"); ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"{name}: Sharpe {o['sharpe_bh']:.2f} -> {o['sharpe_net']:.2f}", fontsize=10)
plt.tight_layout(); plt.show()
print(f"thermostat reaction to last month's return: WTI {ow['react_corr']:+.2f}, "
      f"S&P {os_['react_corr']:+.2f}  (+ = cuts after falls, - = cuts after rallies)")
"""),
        md("""
On stocks the thermostat does what it is famous for: it cuts after falls. On oil it barely
cares which way the price went — it reacts to the *size* of moves, not their direction. It
never cuts after rallies, which is what the mirror would need. And after costs it trails
plain buy-and-hold on the oil tape.
"""),
        md(f"""
## 5 · The verdict

{badges()}

- **Signal: {R['signal']}.** Neither lens sees the mirror. Oil's asymmetry has the *equity*
  sign over 1986-2018 (GJR *t* = {R['wti_gjr_inv_t']:+.2f}), weaker than the S&P's
  ({R['sp_gjr_inv_t']:+.2f}), and it is entirely a post-2008 phenomenon
  (1986-2007 *t* = {R['era_pre_t']:+.2f}; 2008-2018 *t* = {R['era_post_t']:+.2f}).
- **Tradability: {R['trad']}.** The overlay loses Sharpe on the oil tape after costs, and the
  tape is a spot price no one can hold.
- **Mirror? Busted.** The real story is the opposite of the claim: oil went from symmetric to
  stock-like.
"""),
        md("""
## 6 · Could you trade it?

Not from this tape, for three reasons:

1. **Spot is not a position.** WTI spot is the price of a barrel delivered at Cushing today.
   You'd actually hold futures, rolling them monthly — and the roll yield in oil is large
   enough to dominate long-run returns. Nothing here sees it.
2. **There's no edge to harvest.** The mirror would have changed *how you manage oil risk*,
   not given you a return forecast. It isn't there, so nothing changes: on oil, a vol target
   is just a volatility-sizing rule with no special timing benefit.
3. **The tape stops in 2018.** It never saw April 2020, when the front-month WTI future
   settled below zero — a demand crash that would, if anything, make oil look *more* like a
   stock.
"""),
        md("""
## 7 · Going further \U0001F6AA

- **Run it on futures.** A continuous front-month series (with the roll) is the honest
  investable version; the sign question would be the same, the overlay numbers would not.
- **Gold.** The cleanest published case for the inverse effect is gold, which this tape cannot
  test. Our neighbour [993-leverage-effect-asymmetry](../../993-leverage-effect-asymmetry/)
  finds gold close to symmetric.
- **Condition on the shock.** The mirror windows here contain real supply shocks. A test that
  conditions on identified supply vs demand news (rather than on calendar eras) is the natural
  next fork — PRs welcome.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    R = REAL
    cells = [
        md(f"""
# Oil Is Equity in a Mirror — a quantitative teardown \U0001F52C

{badges()}

**Claim.** Equities show the leverage effect (Black 1976; Christie 1982): volatility responds
more to negative than to positive return shocks. Commodities, oil among them, are said to
show the **inverse** asymmetry — volatility responds more to *positive* shocks — because the
shocks that matter are supply disruptions that spike prices. **Test.** GJR-GARCH(1,1,1) and
EGARCH(1,1,1) with robust SEs; a model-free forward-realised-vol regression split by return
sign (Newey-West); a pre-registered two-era split; a rolling 3-year GJR; skewness; a
monthly-average Brent cross-check; a vol-target overlay on WTI and the S&P, one lag, costs.

**Sign convention used everywhere:** the *inverse score* is positive when volatility rises
more after rallies (the claim), negative for the equity sign.

> \U0001F4A1 **In plain words:** we ask the same question four different ways and then ask
> whether the answer would change how you run a volatility-target strategy.
"""),
        code(BOOT),
        md(f"""
## Verdict, up front

| | Stamp | Decisive numbers |
|---|---|---|
| Signal | **{R['signal']}** | WTI 1986-2018 GJR inverse *t* = {R['wti_gjr_inv_t']:+.2f}, EGARCH {R['wti_egarch_inv_t']:+.2f}, model-free {R['wti_mf_t']:+.2f}; eras {R['era_pre_t']:+.2f} / {R['era_post_t']:+.2f} (z of difference {R['era_z']:+.2f}) |
| Tradability | **{R['trad']}** | WTI overlay net Sharpe gain {R['ov_wti_gain']:+.2f} (S&P {R['ov_sp_gain']:+.2f}); spot tape non-investable |
"""),
        md("""
## 1 · Hypotheses (pre-registered in `strategy.verdict`)

- **H1 (Real):** full-sample WTI GJR inverse *t* ≥ 2 **and** model-free *t*(b_up − b_dn) ≥ 2 at
  *h* = 21.
- **H2 (Mixed — regime):** in the pre-registered eras 1986-2007 / 2008-2018, one era has inverse
  *t* ≥ 2 and the other ≤ −2 (or ≥ 2 vs an equity-signed estimate with |z| ≥ 2 on the
  difference).
- **Weak:** one lens or one era clears 2. **None:** otherwise.
- **Tradability:** never Investable (spot). Fragile iff the WTI overlay beats buy-and-hold on
  net excess Sharpe *and* drawdown; else Mirage.
"""),
        md("""
## 2 · The machinery works (synthetic, NOT evidence)

A GJR(1,1,1) with Student-t(8) shocks, 30% vol, persistence 0.97; `signal_strength` k sets
gamma = 0.10·k. k = +1 is equity-type, −1 the mirror, 0 symmetric.
"""),
        code("""
rows = []
for k in (1.0, 0.5, 0.0, -0.5, -1.0):
    px, tr = data.synthetic_gjr(n_years=20, signal_strength=k, seed=1027)
    r = data.log_returns(px)
    g, e = st.fit_asymmetry(r, "gjr"), st.fit_asymmetry(r, "egarch")
    m = st.forward_vol_regression(r, 21)
    rows.append({"k": k, "gamma_true": tr["gamma"], "gamma_hat": g["gamma"],
                 "gjr_inv_t": g["inv_t"], "egarch_inv_t": e["inv_t"], "mf_t": m["diff_t"]})
cal = pd.DataFrame(rows).set_index("k")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cal.index, cal["gjr_inv_t"], "o-", color=OIL, lw=2, ms=8, label="GJR inverse t")
ax.plot(cal.index, cal["mf_t"], "s-", color=EQ, lw=2, ms=8, label="model-free t")
ax.axhspan(-2, 2, color=NULL, alpha=0.15); ax.axhline(0, color="k", lw=1)
ax.set_xlabel("signal_strength k  (+1 equity, -1 mirror)"); ax.set_ylabel("t")
ax.legend(frameon=False); ax.set_title("Both lenses recover sign and size; the null sits in the band")
plt.show(); cal.round(3)
"""),
        md("""
> \U0001F4A1 **In plain words:** when we *plant* the mirror, the tools find it loudly; when we
> plant the stock pattern, they find that with the opposite sign; when we plant nothing, they
> find nothing. So a "no" on real oil is a real "no", not a blind instrument.

## 3 · The teardown on the real tapes

### 3.1 Parametric asymmetry
"""),
        code("""
wti, sp = data.load_wti(), data.load_sp500()
rw, rs = data.log_returns(wti), data.log_returns(sp)
rw99 = rw[rw.index >= sp.index[0]]
rows = []
for lab, r in (("WTI 1986-2018", rw), ("WTI 1999-2018", rw99), ("S&P 1999-2018", rs)):
    for mdl in ("gjr", "egarch"):
        f = st.fit_asymmetry(r, mdl)
        rows.append({"tape": lab, "model": mdl, "gamma": f["gamma"], "robust_se": f["gamma_se"],
                     "inverse_t": f["inv_t"], "persistence": f["persistence"],
                     "converged": f["converged"]})
fits = pd.DataFrame(rows).set_index(["tape", "model"]); fits
"""),
        md("""
GJR `gamma` > 0 is the equity sign, EGARCH `gamma` < 0 is the equity sign; `inverse_t` maps
both onto one scale. **Every row is negative.** WTI's down shocks carry about """
           + f"{R['wti_gjr_ratio']:.2f}" + """× the
variance impact of up shocks (GJR); for the S&P, the fitted up-shock impact is essentially
zero.

> \U0001F4A1 **In plain words:** the standard volatility models say oil, like stocks, gets
> louder after falls — just much less so.

### 3.2 Model-free: forward realised vol split by sign
"""),
        code("""
rows = []
for lab, r in (("WTI 1986-2018", rw), ("WTI 1999-2018", rw99), ("S&P 1999-2018", rs)):
    for h in (5, 21):
        m = st.forward_vol_regression(r, h)
        c = st.sign_correlation(r, h, n_boot=300)
        rows.append({"tape": lab, "h": h, "b_up": m["b_up"], "b_dn": m["b_dn"],
                     "t(b_up-b_dn)": m["diff_t"], "corr": c["corr"], "corr_lo": c["lo"],
                     "corr_hi": c["hi"], "matched_up/down": c["matched_up_over_down"]})
mf = pd.DataFrame(rows).set_index(["tape", "h"])
fig, ax = plt.subplots(figsize=(9, 4))
lab = [f"{a}\\nh={b}" for a, b in mf.index]
ax.bar(range(len(mf)), mf["t(b_up-b_dn)"],
       color=[OIL if "WTI" in a else EQ for a, _ in mf.index], width=0.6)
ax.axhline(2, color="k", ls="--", lw=1); ax.axhline(-2, color="k", ls="--", lw=1)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(len(mf))); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("HAC t of b_up - b_dn"); ax.set_title("Model-free inverse score (+ = the mirror)")
plt.show(); mf.round(3)
"""),
        md("""
`RV(t+1..t+h) = a + b_up·r⁺_t + b_dn·|r⁻_t| + c·RV21(t)`, Newey-West with *h* lags. The full
WTI sample is equity-signed but sub-bar at *h* = 21; 1999-2018 is equity-signed and
significant. The bootstrap correlation interval (blocks ≥ 21 days) agrees in sign.

> \U0001F4A1 **In plain words:** without any model at all, oil after an up day is slightly
> calmer than after a down day of the same size — the stock pattern, faintly.

### 3.3 Property or regime?
"""),
        code("""
eras = {"1986-2007": ("1986-01-01", "2007-12-31"), "2008-2018": ("2008-01-01", "2018-12-31")}
tbl = st.era_table(rw, eras)
d = st.era_difference(tbl, "1986-2007", "2008-2018")
print(f"GJR gamma difference (late - early) {d['diff']:+.4f}, se {d['se']:.4f}, z {d['z']:+.2f}")
roll = st.rolling_asymmetry(rw, window=756, step=63)
roll_sp = st.rolling_asymmetry(rs, window=756, step=63)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(roll.index, roll["inv_t"], color=OIL, lw=2, label="WTI, 3-year GJR inverse t")
ax.plot(roll_sp.index, roll_sp["inv_t"], color=EQ, lw=2, label="S&P 500")
ax.axhspan(-2, 2, color=NULL, alpha=0.15); ax.axhline(0, color="k", lw=1)
ax.axvline(pd.Timestamp("2008-01-01"), color="k", ls=":", lw=1)
ax.set_ylabel("t (window end)"); ax.legend(frameon=False, fontsize=9)
ax.set_title("The mirror flickers in the 1990s; after 2008 oil drifts to the equity side")
plt.show()
print(f"windows with inverse t >= 2: {(roll.inv_t >= 2).mean():.0%}; "
      f"with t <= -2: {(roll.inv_t <= -2).mean():.0%}")
tbl.round(3)
"""),
        md("""
The pre-registered split shows **no** era with the mirror at the bar; the difference between
eras is significant (z ≈ """ + f"{R['era_z']:+.1f}" + """) but it runs *from symmetric to equity-like*.
The rolling windows (overlapping, descriptive, not used for the stamp) show the only
inverse stretches around the 1990-91 Gulf War and the 1996 inventory squeeze.

> \U0001F4A1 **In plain words:** a big supply shock can briefly make rallies the scary
> direction for oil. It is not the normal state, and since 2008 the opposite has been true.

### 3.4 Skewness
"""),
        code("""
wm = data.log_returns(wti.resample("ME").last()); sm_ = data.log_returns(sp.resample("ME").last())
bm = data.log_returns(data.load_crude_monthly()["brent"])
rows = []
for lab, r, blk in (("WTI daily 1986-2018", rw, 21), ("WTI daily 1999-2018", rw99, 21),
                    ("S&P daily 1999-2018", rs, 21), ("WTI month-end", wm, 6),
                    ("S&P month-end", sm_, 6), ("Brent monthly avg", bm, 6)):
    s = st.skew_ci(r, n_boot=300, block=blk)
    rows.append({"series": lab, **s})
sk = pd.DataFrame(rows).set_index("series")
print(st.skew_difference(rw99, rs, n_boot=300))
sk.round(3)
"""),
        md("""
A mirror asset should be positively skewed. WTI is not: its moment skew is negative (driven by
a handful of crash days — 17 January 1991 is −33% in log terms) and its quantile skew is close
to the S&P's; the daily skew difference on common dates is indistinguishable from zero.

### 3.5 Monthly Brent cross-check (monthly *averages* — direction only)
"""),
        code("""
cr = data.load_crude_monthly()
pd.DataFrame({c: st.monthly_asymmetry(cr[c]) for c in cr.columns}).T.round(3)
"""),
        md("""
Same sign as the daily tape, short of the bar. Averaging smooths returns (Working 1960), so
the power here is low by construction.

## 4 · The verdict

The pre-registered rule returns **Signal: """ + R["signal"] + """** (neither lens ≥ 2 on the
inverse side; no era at the bar) and **Tradability: """ + R["trad"] + """** (below).
"""),
        code("""
import json, re
txt = open("../docs/results.md", encoding="utf-8").read()
print(txt[txt.index("## Verdict"):txt.index("---", txt.index("## Verdict"))])
"""),
        md("""
## 5 · Could you trade it? The vol-target overlay

`w_t = min(2, target_t / RV21_t)`, past-only expanding target; **one execution lag**
(weight at close *t* earns *t+1*); costs one-way × |Δw|; excess-vs-excess (S&P minus T-bill;
the WTI spot change treated as a collateralised futures-proxy excess return).
"""),
        code("""
rf = data.load_rf_daily(sp.index)
sw, ss = data.simple_returns(wti), data.simple_returns(sp)
res = {}
for key, r, rfx, c in (("WTI 1987-2018", sw, None, 10.0),
                       ("WTI 1999-2018", sw[sw.index >= sp.index[0]], None, 10.0),
                       ("S&P 1999-2018", ss, rf, 5.0)):
    o = st.vol_target_overlay(r, rfx, cost_bps=c, n_boot=500)
    o.pop("series")
    res[key] = o
ov = pd.DataFrame(res).T[["cost_bps", "sharpe_bh", "sharpe_gross", "sharpe_net", "gain_net",
                          "gain_lo", "gain_hi", "mdd_bh", "mdd_net", "turnover_ann",
                          "react_corr", "breakeven_bps"]]
cs = pd.concat({"WTI": st.cost_sweep(sw, None)["gain_net"],
                "S&P": st.cost_sweep(ss, rf)["gain_net"]}, axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cs.index, cs["WTI"], "o-", color=OIL, lw=2, ms=8, label="WTI spot proxy")
ax.plot(cs.index, cs["S&P"], "s-", color=EQ, lw=2, ms=8, label="S&P 500")
ax.axhline(0, color="k", lw=1); ax.set_xlabel("one-way cost, bp"); ax.set_ylabel("net Sharpe gain")
ax.legend(frameon=False); ax.set_title("The overlay never helps on the oil tape")
plt.show(); ov.astype(float).round(3)
"""),
        md("""
On the S&P the thermostat de-risks after falls (reaction +0.3) and adds a small, statistically
unresolved gain before costs. On WTI the reaction is near zero over the full sample — oil's
volatility responds to the size of a move, not its sign — and the overlay loses Sharpe even
gross. Nowhere does the thermostat de-risk after rallies. Capacity is moot: the tape is spot,
and a real futures book adds a roll the spot ignores.

> \U0001F4A1 **In plain words:** the famous risk-dial behaves on oil the way it does on stocks,
> only more weakly, and on this tape it would have cost you.

## 6 · Going further

- **Futures, with the roll.** The investable version; it would also extend the sample through
  April 2020's negative settlement.
- **Event conditioning.** Replace calendar eras by identified supply vs demand shocks (e.g. a
  structural oil-market VAR decomposition) and test whether the asymmetry follows the *shock
  type* — the mechanism the claim actually proposes.
- **Gold and other commodities**, where the inverse asymmetry has the strongest published
  support; see neighbour study 993 for gold's ratio.
- **Realised-measure GARCH** (HEAVY / realised GARCH) on intraday futures data would sharpen
  the model-free lens considerably.
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
