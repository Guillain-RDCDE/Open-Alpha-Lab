"""Notebook builder for Study 1023 — The Cushing Glut.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    cd notebooks
    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both walk the seven desk beats (METHODOLOGY.md) at two altitudes. Every number quoted in
prose comes from the single ``REAL`` dict below, which mirrors ``docs/results.md`` (re-run
``examples/verify.py`` and update it together). The code cells recompute everything live from
the same frozen, SHA-pinned ``arch`` tape — no network — and label any synthetic cell as such.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# The real run — mirrors docs/results.md (as-of 2020-01-31, fingerprint 4563b3235b4e)
# --------------------------------------------------------------------------- #
REAL = {
    "as_of": "2020-01-31", "fingerprint": "4563b3235b4e", "n_months": 393,
    "signal": "Weak", "trad": "Mirage",
    "break_date": "2010-09", "break_p": "0.002", "sup_f": "766",
    "mean_pre_usd": "−1.41", "mean_post_usd": "+7.69", "shift_usd": "+9",
    "hl_pre": "2.7", "hl_pre_band": "1.9–4.2", "hl_post": "4.2", "hl_post_band": "2.4–12.2",
    "adf_p_full": "0.45", "adf_p_pre": "0.066", "adf_p_pre_usd": "< 0.001",
    "t_gross": "+1.12", "t_net": "+0.84", "ann_net": "+0.94%", "sr_net": "+0.12",
    "sr_gross_ci": "−0.09 to +0.38",
    "t_gross_roll": "+2.09", "t_net_roll": "+1.84",
    "exp_entry": "2009-01", "exp_mae": "−22.4%", "exp_mae_usd": "−25.6", "exp_mae_date": "2011-09",
    "exp_uw": "123",
    "frozen_entry": "2010-05", "frozen_mae": "−23.7%", "frozen_mae_usd": "−25.1",
    "frozen_uw": "114", "frozen_total": "−20.0%",
    "peak_spread": "+27.31", "peak_date": "2011-09",
}

BADGE = {"Real": "2ea44f", "Weak": "dab617", "Mixed": "dab617", "None": "c0392b",
         "Investable": "2ea44f", "Fragile": "dab617", "Mirage": "c0392b"}


def badges() -> str:
    s, t = REAL["signal"], REAL["trad"]
    return (f"![Signal: {s}](https://img.shields.io/badge/Signal-{s}-{BADGE[s]}?style=flat-square)\n"
            f"![Tradability: {t}](https://img.shields.io/badge/Tradability-{t}-{BADGE[t]}"
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
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
BLUE, RED, GREEN, GREY, AMBER = "#1f6feb", "#c0392b", "#2ea44f", "#8b949e", "#b08800"
from cushing import data, strategy as st
px = data.load_crude()
sf = st.spread_frame(px)
print(f"real tape: arch crude, {len(px)} months {px.index[0]:%Y-%m} → {px.index[-1]:%Y-%m}, "
      f"as-of {data.AS_OF}, fingerprint {data.fingerprint(px)}")
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
# The Cushing Glut 🛢️
### Brent and WTI are the same oil in two places. Until the tanks in Oklahoma filled up.

{badges()}

Traders have long faded the gap between the world's two benchmark crudes: same light sweet
oil, so the price difference should be about the cost of shipping it, and any stretch should
snap back. For twenty-three years of this tape it nearly did. Then, in {R['break_date']}, the
gap found a new home about ${R['shift_usd'][1:]} a barrel away, and anyone who kept fading it
back to the old one spent most of a decade underwater.

> 📓 **This is the plain-language layer.** The unit-root tests, the break bootstrap and the
> power study are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> ⚠️ **Not investment advice.** The prices here are **spot, monthly averages** — you cannot
> trade them. A real trade uses futures and pays a monthly roll, so every profit shown is an
> **upper bound**.
"""),
        md(f"""
## Beat 0 · The verdict (real tape)

| | Stamp | In one line |
|---|---|---|
| **Signal** | **{R['signal']}** | the gap *was* tethered — reverting with a half-life of about {R['hl_pre']} months — then a data-chosen break in {R['break_date']} moved its home; across the whole sample it is not tethered |
| **Tradability** | **{R['trad']}** | fading it in real time earned {R['ann_net']} a year after costs (t = {R['t_net']}) on prices nobody can trade; the trader who learned the tether before 2010 was underwater {R['frozen_uw']} months |

*Numbers from [`docs/results.md`](../docs/results.md) (as-of {R['as_of']}, fingerprint
`{R['fingerprint']}`); the charts below are recomputed live from the same pinned tape.*
"""),
        code(BOOT),
        md("""
## 1 · The claim

> *"Brent and WTI are the same commodity in two places. The spread is tethered by transport
> cost and always mean-reverts — fade it when it stretches."*

It is a good claim. Both are light, sweet crudes that refineries can largely swap. If WTI
gets too cheap, someone buys it, ships it and sells it as Brent-priced oil; if Brent gets too
cheap, the reverse. That physical arbitrage is the rubber band.
"""),
        md("""
## 2 · So what?

If it were true, it would be one of the cleanest trades in commodities: a bet on two
near-identical barrels converging, with no view on the price of oil itself. It would also
say something comforting about markets — that physical arbitrage keeps prices honest.

The catch the claim skips: the rubber band is only as strong as the *pipes*. Arbitrage needs
somewhere to move the oil. Cushing, Oklahoma — where WTI is delivered — is a landlocked tank
farm. When US shale output surged around 2010 faster than pipelines out of Cushing were built,
the oil had nowhere to go.
"""),
        md("""
## 3 · How we'd know

Before running anything we wrote down what would count:

- **Tethered "always"** means the gap is stationary over the *whole* sample, there is no
  significant break in its average level, and a fader using only past data makes money with a
  robust t-statistic of at least 2.
- **Tethered within a regime** would mean a significant break, with the gap stationary before it.
- **Tradable** needs the real-time fader to survive costs (t ≥ 2) *and* the trader who
  calibrated on pre-2010 data to recover within three years.

The break date is **found by the data** (a scan over every possible month), not picked by hand.
"""),
        md("## 4 · The teardown\n\n### 4.1 The gap, month by month"),
        code("""
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(sf.index, sf["spread"], color=BLUE, lw=1.6, label="Brent − WTI ($/bbl)")
ax.axhline(0, color="k", lw=0.8)
b = st.sup_f_mean(sf["log_ratio"])
ax.axvline(b["break_date"], color=RED, ls="--", lw=1.5,
           label=f"break found by the data: {b['break_date']:%Y-%m}")
pre = sf["spread"][sf.index < b["break_date"]].mean()
post = sf["spread"][sf.index >= b["break_date"]].mean()
ax.hlines(pre, sf.index[0], b["break_date"], color=GREY, lw=2, label=f"average before: {pre:+.1f}")
ax.hlines(post, b["break_date"], sf.index[-1], color=AMBER, lw=2, label=f"average after: {post:+.1f}")
ax.set_ylabel("$ per barrel"); ax.legend(fontsize=9, loc="upper left")
ax.set_title("For two decades WTI sat about a dollar above Brent. Then it didn't.", fontsize=11)
plt.show()
"""),
        md(f"""
Before the break, WTI usually traded a little **above** Brent (about $1.4 on average).
After it, Brent sat about **$7.7 above** WTI on average, peaking near ${R['peak_spread'][1:]} in
{R['peak_date']}. The gap still wiggled and snapped back — around a *different* centre.

> 🔬 **For the quants.** The break comes from a sup-F scan for a single mean shift, with a
> bootstrap p-value of {R['break_p']} against a persistent, break-free AR(1) — the textbook
> critical values are far too lenient for a series this sticky.
"""),
        md("### 4.2 How fast did it snap back?"),
        code("""
hl = st.halflife_by_regime(sf["log_ratio"], [b["break_date"]])
show = hl[["start", "end", "n", "halflife", "halflife_lo", "halflife_hi"]].copy()
show["start"] = show["start"].dt.strftime("%Y-%m"); show["end"] = show["end"].dt.strftime("%Y-%m")
show.columns = ["from", "to", "months", "half-life (months)", "low", "high"]
show
"""),
        md(f"""
A **half-life** is how long it takes for half of a stretch to disappear. Before the break,
about {R['hl_pre']} months — a fast rubber band. After it, about {R['hl_post']} months around
the new level, with a much wider range of uncertainty. Inside each era, the claim looks
fine. The problem is the jump between them.
"""),
        md("### 4.3 What the fader actually lived through"),
        code("""
rules = {"real-time (expanding window)": st.zscore_realtime(sf["log_ratio"]),
         "frozen on 1987–2009": st.zscore_frozen(sf["log_ratio"])}
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for (name, z), col in zip(rules.items(), (BLUE, RED)):
    bk = st.book(px, st.positions(z))
    eq = (1 + bk["net"]).cumprod()
    axes[0].plot(eq.index, eq, color=col, lw=1.8, label=name)
    axes[1].plot(bk.index, bk["pos"], color=col, lw=1.4, drawstyle="steps-post", label=name)
axes[0].axvline(b["break_date"], color=GREY, ls="--", lw=1)
axes[0].set_ylabel("growth of $1 (net, spot)"); axes[0].legend(fontsize=9)
axes[1].set_ylabel("position (−1 = short the spread)"); axes[1].set_yticks([-1, 0, 1])
axes[1].legend(fontsize=9)
axes[0].set_title("Fading the gap: a slow grind up, then a decade of waiting", fontsize=11)
plt.tight_layout(); plt.show()
"""),
        md(f"""
Both traders bet that the gap would close — "short the spread" means betting Brent falls back
toward WTI. The real-time trader was already short from {R['exp_entry']} and, at the worst
point ({R['exp_mae_date']}), was down **{R['exp_mae']}** of the money behind the trade, or about
**${R['exp_mae_usd'][1:]} a barrel**. The trader who learned the "normal" gap on 1987–2009 data
went short in {R['frozen_entry']} and was **still underwater {R['frozen_uw']} months later**
when the tape ends — the gap never went back to its old home.
"""),
        md(f"""
## 5 · The verdict

**Signal: {R['signal']}.** The rubber band was real *within* each era — before {R['break_date']}
the gap reverted with a {R['hl_pre']}-month half-life — but the claim says *always*, and over the
whole tape the gap is not stationary (ADF p = {R['adf_p_full']}). A data-found break moved the
gap's centre by about ${R['shift_usd'][1:]}. Fading it in real time earned a gross t of
{R['t_gross']}, short of the desk's bar of 2.

**Tradability: {R['trad']}.** After costs the real-time fader made {R['ann_net']} a year
(Sharpe {R['sr_net']}, t = {R['t_net']}) — and that is on spot prices that can't be traded, so
it is the *best case*. The pre-2010 believer lost about {R['frozen_mae'][1:]} at the worst and
never got back to even.
"""),
        md(f"""
## 6 · Could you trade it?

Not as stated. A real trader holds ICE Brent and NYMEX WTI **futures** and rolls them every
month. During the glut, WTI futures were in steep *contango* (later months dearer than the
front), so being long WTI — the side the fader takes — meant **paying** to roll on top of the
losses above. The spot tape cannot see that cost, which is why the numbers here are an upper
bound. A rolling 60-month window adapts faster (gross t {R['t_gross_roll']}), but picking the
window that worked after the fact is not a strategy.
"""),
        md("""
## 7 · Going further 🚪

- **Use the futures.** Rebuild the trade on rolled ICE Brent / NYMEX WTI contracts and charge
  the actual roll yield — the number most likely to make the post-2010 loss worse.
- **Watch the pipes.** The gap's home is set by transport capacity: Seaway's reversal (2012),
  the 2015 end of the US crude-export ban, new Gulf Coast pipelines. A model that moves the
  mean with infrastructure, not a fixed z-score, is the honest version of the claim.
- **Fork it:** change the entry/exit bands or the window in `cushing/strategy.py` and re-run
  `examples/verify.py` — and remember that trying many and reporting the best is the trap.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(f"""
# The Cushing Glut — a quantitative teardown 🔬
### ADF + KPSS · Engle–Granger · bootstrapped sup-F / CUSUM / Bai–Perron · OU half-lives · a real-time fader

{badges()}

The companion to the [notebook for the curious](01_for_the_curious.ipynb). The load-bearing
results: **§3** (full-sample stationarity fails), **§4** (a mean break at {R['break_date']},
bootstrap p = {R['break_p']}), **§6** (the real-time fader, gross HAC t {R['t_gross']}) and
**§7** (power: why a level break looks like a unit root).

> ⚠️ **Not investment advice.** Real tape: monthly **spot** Brent and WTI, **calendar-month
> averages** (verified against the daily FRED WTI tape), nominal US$, shipped in `arch` and
> SHA-256 pinned via `quantlab.bundled`. Futures roll yield is **not** in any P&L here —
> every backtest is an **upper bound**. Synthetic cells are labelled *synthetic*.
>
> 💡 **The `💡 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT),
        md(f"""
## Verdict, up front (real tape — from [`docs/results.md`](../docs/results.md))

| Axis | Stamp | Decisive numbers |
|---|---|---|
| **Signal** | {R['signal']} | full-sample log ratio ADF p = {R['adf_p_full']}, KPSS rejects; sup-F break {R['break_date']} (bootstrap p = {R['break_p']}); pre-break half-life {R['hl_pre']} m ({R['hl_pre_band']}), pre-break ADF p = {R['adf_p_pre']} in logs ({R['adf_p_pre_usd']} in $); real-time fader gross HAC t {R['t_gross']}, block-bootstrap SR CI {R['sr_gross_ci']} |
| **Tradability** | {R['trad']} | net {R['ann_net']}/yr, SR {R['sr_net']}, HAC t {R['t_net']} on untradeable spot; frozen pre-2010 trader MAE {R['frozen_mae']} ({R['frozen_mae_usd']} $/bbl), underwater {R['frozen_uw']} months, never recovered |

The pre-registered rule (`strategy.verdict`) would have read **Mixed** (tethered within regime)
had the pre-break log ratio cleared ADF at 5%; it misses at p = {R['adf_p_pre']} — the
pre-break regime carries a smaller shift of its own in the mid-2000s. In dollars it clears
easily. The stamp follows the rule, and the rule reads logs.

> 💡 **In plain words.** The tether was real inside each era and absent across them.
"""),
        md("""
## 1 · Hypotheses (fixed before the run)

- **H₁ (stationary).** The log Brent/WTI ratio is stationary over the full sample (ADF rejects
  *and* KPSS does not).
- **H₂ (no break).** A sup-F scan for a mean shift is insignificant against a persistent,
  break-free AR(1) bootstrap null.
- **H₃ (fade pays).** A z-score fader using only past data (expanding window, ±2σ in, ±0.5σ out,
  one lag) earns a gross HAC t ≥ 2.
- **H₄ (survivable).** Net HAC t ≥ 2, and a trader calibrated on 1987–2009 is underwater ≤ 36
  months.

Real requires H₁–H₃; Mixed requires a break plus a stationary pre-break regime; Fragile
requires H₄. Investable is unreachable on a spot tape by construction.
"""),
        md("## 2 · The tape — and why it is not investable"),
        code("""
chk = data.check_monthly_average(px)
print(chk)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(px.index, px["brent"], color=BLUE, lw=1.4, label="Brent spot (monthly avg)")
ax.plot(px.index, px["wti"], color=AMBER, lw=1.4, label="WTI spot (monthly avg)")
ax.set_ylabel("US$/bbl, nominal"); ax.legend(fontsize=9)
plt.show()
"""),
        md("""
> 💡 **In plain words.** The monthly WTI number matches the average of daily FRED prints to a
> quarter of a cent — so each point is a month's *average*, a price nobody could deal at. That
> smoothing also adds a little spurious reversion to monthly changes (Working 1960).
"""),
        md("## 3 · Stationarity: ADF and KPSS, full sample and sub-samples"),
        code("""
b = st.sup_f_mean(sf["log_ratio"])
bd = b["break_date"]
pre_end = (bd - pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
windows = [("full sample", None, None), ("1987–2003", None, "2003-12-31"),
           ("1987–2009", None, "2009-12-31"), ("pre-break", None, pre_end),
           ("post-break", bd, None), ("2010–2014", "2010-01-31", "2014-12-31"),
           ("2015–2020", "2015-01-31", None)]
stab = st.stationarity_table(sf, windows)
stab[["window", "series", "n", "adf_stat", "adf_p", "kpss_stat", "kpss_p", "reading"]]
"""),
        code("""
eg = pd.DataFrame([{"window": w, **st.engle_granger(px, a, z)} for w, a, z in windows])
eg[["window", "n", "beta", "stat", "p", "reject5"]]
"""),
        md("""
> 💡 **In plain words.** Over the whole sample the 1:1 spread is not stationary by either test.
> Engle–Granger still rejects no-cointegration on the full sample — but only by choosing a
> slope above one, which is not the claim's "same commodity" spread; over 2010–2014 alone it
> cannot reject at all.
"""),
        md("## 4 · Locating the break — by the data, not by hand\n\n### 4.1 sup-F with a bootstrap null"),
        code("""
boot = st.sup_f_bootstrap(sf["log_ratio"], n_boot=199)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
axes[0].plot(b["F_path"].index, b["F_path"], color=BLUE, lw=1.6)
axes[0].axvline(bd, color=RED, ls="--", lw=1.3, label=f"argmax {bd:%Y-%m}")
axes[0].axhline(st.ANDREWS_CV_15["5%"], color=GREY, lw=1, label="Andrews 5% (i.i.d.-ish errors)")
axes[0].axhline(boot["null_q95"], color=AMBER, lw=1.5, label="bootstrap 95% (persistent null)")
axes[0].set_ylabel("F statistic, mean shift at date"); axes[0].legend(fontsize=8)
c = st.cusum_mean(sf["log_ratio"])
axes[1].plot(c["path"].index, c["path"], color=BLUE, lw=1.6, label="HAC-scaled CUSUM")
for s_ in (1, -1):
    axes[1].axhline(s_ * st.CUSUM_CV_5, color=GREY, ls="--", lw=1)
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"sup-F {boot['sup_f']:.0f} at {boot['break_date']:%Y-%m}; bootstrap p {boot['p']:.3f} "
      f"(null phi {boot['null_phi']:.3f}, null 95%/99% {boot['null_q95']:.0f}/{boot['null_q99']:.0f})")
print(f"CUSUM {c['stat']:.2f} (5% cv {c['cv5']}), peak {c['peak_date']:%Y-%m}")
"""),
        md(f"""
> 💡 **In plain words.** A sticky series produces big F statistics by itself, so the
> asymptotic critical value (8.7) is meaningless here; the bootstrap asks how big the biggest
> F gets in a *break-free* world with the same stickiness. The real one ({R['sup_f']}) is
> larger than almost all of them. (The notebook uses 199 replications for speed; `verify.py`
> uses 499.)
"""),
        md("### 4.2 Bai–Perron: several breaks at once"),
        code("""
bp = st.bai_perron(sf["log_ratio"], max_breaks=5, min_seg=24)
print(f"LWZ picks {bp['m_lwz']} breaks; BIC picks {bp['m_bic']}")
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(sf.index, sf["log_ratio"], color=BLUE, lw=1.2, label="log(Brent/WTI)")
for s_ in bp["segments"]:
    ax.hlines(s_["mean"], s_["start"], s_["end"], color=RED, lw=2.2)
ax.plot([], [], color=RED, lw=2.2, label="Bai–Perron segment means (LWZ)")
ax.legend(fontsize=9); plt.show()
bp["table"]
"""),
        md("""
> 💡 **In plain words.** Information criteria over-count breaks when errors are this
> persistent (the test-suite shows LWZ "finding" breaks in a break-free synthetic tether), so
> the count is descriptive. What every method agrees on is the big shift around 2010/2011.
"""),
        md("## 5 · Half-lives before and after"),
        code("""
rows = []
for col in ("log_ratio", "spread"):
    t = st.halflife_by_regime(sf[col], [bd]); t.insert(0, "series", col); rows.append(t)
hl = pd.concat(rows, ignore_index=True)
hl["start"] = hl["start"].dt.strftime("%Y-%m"); hl["end"] = hl["end"].dt.strftime("%Y-%m")
hl
"""),
        md(f"""
> 💡 **In plain words.** Before the break a stretch halved in ~{R['hl_pre']} months; after, ~{R['hl_post']}
> months around a higher mean, with a band out to {R['hl_post_band'].split('–')[1]}. OLS φ is
> biased down in short samples, so both lean fast. In logs the pre-break ADF p is
> {R['adf_p_pre']}; in dollars it is {R['adf_p_pre_usd']}.
"""),
        md("## 6 · The trader's experience — real time, one lag, with costs"),
        code("""
rules = {"expanding": st.zscore_realtime(sf["log_ratio"]),
         "rolling 60m": st.zscore_realtime(sf["log_ratio"], window=60),
         "frozen 1987–2009": st.zscore_frozen(sf["log_ratio"])}
rows, books = [], {}
for name, z in rules.items():
    bk = st.book(px, st.positions(z)); books[name] = bk
    g, n = st.perf(bk["gross"], n_boot=500), st.perf(bk["net"], n_boot=500)
    uw = st.underwater(bk["net"], "2010-01-31" if name.startswith("frozen") else None)
    rows.append({"rule": name, "gross /yr": g["ann"], "gross t": g["t_hac"],
                 "SR gross": g["sharpe"], "SR lo": g["sr_lo"], "SR hi": g["sr_hi"],
                 "net /yr": n["ann"], "net t": n["t_hac"], "maxDD": n["maxdd"],
                 "underwater m": uw["months"]})
pd.DataFrame(rows).set_index("rule")
"""),
        code("""
fig, ax = plt.subplots(figsize=(11, 4.6))
for (name, bk), col in zip(books.items(), (BLUE, GREEN, RED)):
    ax.plot(bk.index, (1 + bk["net"]).cumprod(), color=col, lw=1.6, label=name)
ax.axvline(bd, color=GREY, ls="--", lw=1)
ax.set_ylabel("growth of $1, net (spot: an upper bound)"); ax.legend(fontsize=9)
plt.show()
ep = st.episodes(books["frozen 1987–2009"])
pd.concat([st.episodes(books["expanding"]).assign(rule="expanding"), ep.assign(rule="frozen")])
"""),
        md(f"""
> 💡 **In plain words.** The expanding-window fader's worst trade opened in {R['exp_entry']}
> and was down {R['exp_mae']} ({R['exp_mae_usd']} $/bbl) at the {R['exp_mae_date']} peak of the
> glut. The frozen pre-2010 trader went short in {R['frozen_entry']} and was still in the trade,
> {R['frozen_total']} net, when the tape ended {R['frozen_uw']} months into its drawdown. The
> rolling window looks best (gross t {R['t_gross_roll']}) because it forgets faster — but it is
> one of many windows, chosen with hindsight, and still below 2 net.
"""),
        code("""
cs = st.cost_sweep(px, st.positions(rules["expanding"]))
print("break-even one-way cost per leg (bp):",
      round(st.break_even_cost(px, st.positions(rules["expanding"]))))
sub = []
for lab, a, z in [("1987–2009", None, "2009-12-31"), ("2010–2014", "2010-01-31", "2014-12-31"),
                  ("2015–2020", "2015-01-31", None)]:
    bk = st._window(books["expanding"], a, z)
    sub.append({"period": lab, "gross t": st.perf(bk["gross"], n_boot=200)["t_hac"],
                "net t": st.perf(bk["net"], n_boot=200)["t_hac"]})
display(cs); pd.DataFrame(sub).set_index("period")
"""),
        md("""
> 💡 **In plain words.** Costs are not what kills it — the break-even cost is large because
> the rule trades rarely. What kills it is that the edge itself is thin (t ≈ 1) and the losses
> concentrate in one episode the rule could not see coming. With n ≈ 390 months and six
> round-trips, the sample simply cannot certify a mean-reversion premium.
"""),
        md("## 7 · Power — what n = 393 can see (synthetic)"),
        code("""
# SYNTHETIC: OU log ratios calibrated to the pre-break regime (half-life 3 m at strength 1)
pw = st.power_curve(strengths=(0.0, 0.1, 0.25, 0.5, 1.0), n_reps=40)
pwb = st.power_curve(strengths=(0.0, 0.1, 0.25, 0.5, 1.0), n_reps=40, break_size=0.15)
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(pw.index, pw["adf_reject"], "o-", color=BLUE, lw=2, label="ADF rejects, no break")
ax.plot(pwb.index, pwb["adf_reject"], "s--", color=RED, lw=2, label="ADF rejects, one planted break")
ax.plot(pwb.index, pwb["kpss_reject"], "^:", color=AMBER, lw=2, label="KPSS rejects, one planted break")
ax.axhline(0.05, color=GREY, lw=1)
ax.set_xlabel("signal_strength (0 = random walk, 1 = 3-month half-life)")
ax.set_ylabel("rejection rate at 5%"); ax.legend(fontsize=9)
plt.show()
pw.join(pwb.add_suffix("_break"))
"""),
        md("""
> 💡 **In plain words.** At this sample length ADF finds a three-month tether every time — so
> the real full-sample failure is not a power problem. But plant one level shift the size of
> the real one and ADF mostly stops rejecting while KPSS rejects nearly always: exactly the
> real tape's signature. A broken tether *looks like* no tether (Perron 1989).
"""),
        code("""
# SYNTHETIC: the break test itself — quiet without a break, loud with one
for bsz in (0.0, 0.15):
    spx, tr = data.synthetic_spread(signal_strength=1.0, break_size=bsz, seed=7)
    q = st.sup_f_bootstrap(np.log(spx["brent"] / spx["wti"]), n_boot=199)
    print(f"planted break {bsz:.2f}: bootstrap p = {q['p']:.3f}, located {q['break_date']:%Y-%m}"
          f" (true {tr['break_date']:%Y-%m})" if bsz else
          f"planted break {bsz:.2f}: bootstrap p = {q['p']:.3f}")
tp = st.synthetic_trade_power(strengths=(0.0, 0.5, 1.0), n_reps=20)
tp
"""),
        md(f"""
## 8 · The verdict

**Signal: {R['signal']}.** H₁ fails (full-sample ADF p = {R['adf_p_full']}, KPSS rejects). H₂
fails (break at {R['break_date']}, bootstrap p = {R['break_p']}; the dollar spread's mean moved
from {R['mean_pre_usd']} to {R['mean_post_usd']} $/bbl). H₃ fails (gross HAC t {R['t_gross']},
block-bootstrap SR CI {R['sr_gross_ci']}). The Mixed branch misses narrowly in logs. There is
mean reversion here — fast, within each regime — but the claim's "always" is what the tape
rejects.

**Tradability: {R['trad']}.** H₄ fails on both legs: net t {R['t_net']} and {R['frozen_uw']}
months underwater for the pre-2010 believer, on a tape that flatters the trade (spot, monthly
averages, no roll yield).

## 9 · Could you trade it?

The executable version is a calendar-matched ICE Brent vs NYMEX WTI futures spread (or the
exchange-listed Brent–WTI spread contracts). Per-leg transaction costs are small and barely
matter (see the cost sweep). The roll does: from 2009 to 2011 the WTI curve was in deep
contango as Cushing filled, so the fader's long-WTI leg paid to roll every month on top of the
spot loss. Capacity is not the binding constraint — both legs are among the deepest commodity
futures in the world. The binding constraint is that the anchor is infrastructure, and
infrastructure changed.

## 10 · Going further

- **Futures-based rebuild** with actual roll yields — the obvious next PR.
- **A moving anchor.** Model the mean as a function of Cushing inventories or pipeline capacity
  (EIA data) and test whether reversion to *that* is stable — the honest form of the claim.
- **Daily data** would raise n and the power of every test here; the bundled daily tape has WTI
  only, so it is not attempted.
- **Regime-switching nulls** for the break bootstrap (heavier tails, volatility regimes) would
  widen the null; the AR(1) null used here is the simplest defensible one.
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
