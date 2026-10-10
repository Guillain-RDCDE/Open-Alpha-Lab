"""Notebook builder for Study 1014 — The Perfect Macro Forecaster.

The two narrative notebooks are a *generated artefact*: edit the cell text here, then

    python build_notebooks.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both walk the same seven desk beats (METHODOLOGY.md) at two altitudes. The code cells run on
the real bundled tapes (offline, SHA-pinned), so every figure is the real thing. The numbers
quoted in the prose live in ONE place — the ``REAL`` dict below — and mirror
``docs/results.md``; rerun ``examples/verify.py`` and update this dict together.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# The real-tape headline — mirrors docs/results.md (as-of 2009-09-30, fp 6008ef31746a)
# --------------------------------------------------------------------------- #
REAL = {
    "as_of": "2009-09-30", "fingerprint": "6008ef31746a",
    "window": "1964Q4–2009Q1", "n_q": 178,
    "signal": "None", "trad": "Fragile", "third": "Confirmed",
    "bh": 0.25, "ceiling": 1.53,
    "head": 0.29, "head_d": "+0.04", "head_boot": 0.40, "head_rot": 0.12, "head_cap": "3%",
    "lead": 0.47, "lead_d": "+0.22", "lead_boot": 0.09, "lead_rot": 0.01, "lead_cap": "17%",
    "g_head": 0.21, "g_lead": 0.58, "g_lead_boot": 0.02, "g_lead_rot": 0.006,
    "real": 0.48, "real_d": "+0.23", "real_boot": 0.08, "real_rot": 0.006,
    "strict": 0.13, "strict_d": "−0.12",
    "ll0": "+0.07", "ll0_t": "+0.8", "ll1": "+0.27", "ll1_t": "+3.6", "ll2": "+0.34",
    "ll2_t": "+4.4",
    "contrast_g": "+0.37", "contrast_g_p": 0.004,
    "one_sentence": (
        "Even a flawless forecast of the coming quarter's GDP, jobs and inflation would have "
        "timed US stocks to a Sharpe of 0.29 against 0.25 for buying and holding, because the "
        "market had already moved on to pricing the quarter after — only an oracle that sees "
        "that far (0.47) starts to earn its fee; the lagged rule a real investor could run "
        "looks better on paper (0.48) but loses to buy-and-hold with one more quarter of lag "
        "(0.13)."),
}

COLOR = {"good": "#2ea44f", "amber": "#dab617", "bad": "#c0392b", "blue": "#1f6feb",
         "grey": "#8b949e"}

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Stocks lead the economy?: Confirmed](https://img.shields.io/badge/Stocks_lead_the_economy%3F-Confirmed-8b949e?style=flat-square)")

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/ lives there)
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
GOOD, AMBER, BAD, BLUE, GREY = "#2ea44f", "#dab617", "#c0392b", "#1f6feb", "#8b949e"

from clairvoyance import data, strategy as st
panel = data.load_panel()
print(f"as-of {data.AS_OF} | {len(panel)} quarters {panel.index[0].date()} -> "
      f"{panel.index[-1].date()} | fingerprint {data.fingerprint(panel[['mkt','rf','g','du','infl','dtb']])}")
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
# The Perfect Macro Forecaster \U0001F52E
### Suppose you knew the economy's next move before anyone else. How rich would you get?

{BADGES}

A whole industry forecasts GDP, jobs and inflation, on the premise that the economy is where
stocks are heading. We skip the hard part and hand an investor a **perfect** forecast — the
real, revised numbers, delivered in advance — then ask what it is worth for timing the US stock
market. The answer is: almost nothing, unless the crystal ball sees *further* than the
forecasters do.

> \U0001F4D3 **This is the plain-language layer.** The bootstrap, the rotation placebo and every
> cell of the oracle grid are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
> Headline numbers come from the pinned real run in [`docs/results.md`](../docs/results.md)
> (as-of {R['as_of']}, fingerprint `{R['fingerprint']}`); every chart below is drawn live from
> the same bundled tapes.
>
> ⚠️ **Not investment advice.** Research and education.
"""),
        code(BOOT),
        md(f"""
## Beat 0 · The answer first

| Question | Answer |
|---|---|
| Knowing **this coming quarter's** GDP, jobs and inflation perfectly — does it beat buying and holding? | **No.** Sharpe {R['head']:.2f} vs {R['bh']:.2f}, well inside luck. |
| Knowing the quarter **after** that? | It starts to help: {R['lead']:.2f} (GDP alone {R['g_lead']:.2f}). |
| Why? | Stocks **lead** the economy by one to two quarters. They price the future. |
| What does a real investor — who only sees *last* quarter's data — get? | {R['real']:.2f} on paper, but {R['strict']:.2f} with one more quarter of lag. Fragile. |
| And perfect knowledge of the market itself? | {R['ceiling']:.2f}. Macro clairvoyance captures about {R['head_cap']} of that. |
"""),
        md("""
## Beat 1 · The claim

*"If you knew where GDP, unemployment and inflation were heading, you would know where stocks
are heading."* It is the unspoken premise of every quarterly outlook, every recession call and
every "economy is strong, so stay invested" note. It sounds like common sense: company profits
come from the economy, so a better economy should mean higher stock prices.

We steelman it as hard as possible. No forecasting error at all: our investor gets the
**final, revised** numbers — the ones economists only agree on years later — *before* the
quarter starts.
"""),
        md("""
## Beat 2 · So what?

If the premise were right, the macro-forecasting industry would be sitting on a money machine,
and a perfect forecaster would sidestep every bear market. If it is wrong, then even flawless
economic forecasts are the wrong tool for timing stocks — and the reason why says something
deep about markets: **they do not wait for the data.**
"""),
        md("""
## Beat 3 · How we'd know

Every quarter from 1964 to 2009 the investor is either fully in US stocks (total return,
dividends included) or fully in Treasury bills. The oracle reveals one number, and the
investor holds stocks only when that number is on the "good" side of its own history
(above-median growth, below-median unemployment change or inflation).

The key dial is **how far ahead the oracle sees**, relative to the quarter you hold:

- **h = 0** — the quarter you are about to hold. A perfect version of what forecasters sell.
- **h = +1, +2** — one or two quarters *beyond* it.
- **h = −1, −2** — the last published data. What a real investor actually has.

We announce the mirage line up front: if the h = 0 oracle cannot beat buying and holding beyond
what luck produces, the claim fails.

> \U0001F52C **For the quants.** Luck is measured two ways: a block bootstrap of the Sharpe
> difference, and a rotation placebo that slides the oracle's own signal against the returns.
"""),
        md("""
## Beat 4 · The teardown

### 4.1 Who moves first — stocks or the economy?
"""),
        code("""
L = st.lead_lag(panel, "g")
fig, ax = plt.subplots(figsize=(10, 4.8))
cols = [GOOD if t > 2 else GREY for t in L["t"]]
ax.bar(L.index, L["corr"], color=cols, width=0.6)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(L.index)
ax.set_xticklabels([f"{k:+d}" for k in L.index])
ax.set_xlabel("GDP growth measured k quarters AFTER the stock-market quarter")
ax.set_ylabel("correlation with the quarter's stock return")
ax.set_title("Stocks move first: they correlate with GDP one to two quarters LATER "
             "(green = significant)", fontsize=10)
plt.show()
print(L[["corr", "t"]].round(2).T.to_string())
"""),
        md("""
The tall green bars sit to the *right* of zero. The stock market's return this quarter lines up
with the economy's growth **one and two quarters later** — and barely at all with the quarter it
is living through. By the time GDP happens, the market has already moved.
"""),
        md("""
### 4.2 What a perfect forecast is worth, by how far it sees
"""),
        code("""
T, win = st.horizon_table(panel, n_boot=500, with_rotation=False)
sub = panel.loc[win]
bh = st.ann_sharpe(sub["ex"])
piv = T.pivot(index="h", columns="var", values="sharpe")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(piv.index, piv["g"], "o-", lw=2, color=BLUE, label="GDP growth oracle")
ax.plot(piv.index, piv["combined"], "s-", lw=2, color=GOOD, label="combined oracle (GDP + jobs + inflation)")
ax.axhline(bh, color="k", ls="--", lw=1.5, label=f"buy and hold ({bh:.2f})")
ax.axvline(0, color=GREY, lw=1)
ax.annotate("perfect forecast of\\nthe quarter you hold", (0, piv.loc[0, "g"]),
            xytext=(0.15, 0.05), fontsize=9)
ax.set_xticks(piv.index)
ax.set_xlabel("how far ahead the oracle sees (quarters, relative to the holding quarter)")
ax.set_ylabel("Sharpe ratio (net of costs)")
ax.legend(fontsize=9)
plt.show()
print(piv.round(2).to_string())
"""),
        md(f"""
At h = 0 — a flawless forecast of the very quarter you are about to invest in — both oracles
sit on the buy-and-hold line (the combined one scores {R['head']:.2f} against {R['bh']:.2f}). Move
the crystal ball one quarter further out and the GDP oracle jumps to {R['g_lead']:.2f}. The
economy you need to know about is not the one forecasters forecast; it is the one *after* it.
"""),
        md("""
### 4.3 Growing a dollar
"""),
        code("""
P = st.all_positions(panel).loc[win]
curves = {
    "buy and hold": sub["ex"] + sub["rf"],
    "perfect forecast of the coming quarter (combined, h=0)": st.book(P["combined@+0"], sub["ex"]) + sub["rf"],
    "oracle one quarter beyond (combined, h=+1)": st.book(P["combined@+1"], sub["ex"]) + sub["rf"],
    "last quarter's data (combined, h=-1)": st.book(P["combined@-1"], sub["ex"]) + sub["rf"],
}
colors = ["k", BLUE, GOOD, AMBER]
fig, ax = plt.subplots(figsize=(10, 5.2))
for (name, r), c in zip(curves.items(), colors):
    ax.plot((1 + r).cumprod(), lw=2, color=c, label=name)
ax.set_yscale("log")
ax.set_ylabel("growth of $1 (log scale, total return)")
ax.legend(fontsize=8.5)
plt.show()
"""),
        md("""
> \U0001F52C **For the quants.** These are total-return wealth curves (dividends in, bills
> earned when out of the market). Over 45 years a Sharpe gap of 0.2 looks large on a log chart;
> whether it is distinguishable from luck is the bootstrap's job, in notebook 02.
"""),
        md("""
### 4.4 The ceiling — what if the oracle knew the market itself?
"""),
        code("""
lad = st.ceiling_ladder(panel, win)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(lad["hit_rate"] * 100, lad["sharpe"], "o-", color=GREY, lw=2,
        label="market oracle, right this % of quarters")
for name, c in (("combined@+0", BLUE), ("combined@+1", GOOD), ("g@+1", AMBER)):
    r = T[(T["var"] + "@" + T["h"].map(lambda x: f"{x:+d}")) == name].iloc[0]
    ax.scatter(r["hit_rate"] * 100, r["sharpe"], s=90, color=c, zorder=3,
               label=f"macro oracle {name}")
ax.axhline(bh, color="k", ls="--", lw=1.2, label="buy and hold")
ax.set_xlabel("% of quarters the oracle calls the market's direction right")
ax.set_ylabel("Sharpe ratio")
ax.legend(fontsize=8.5)
plt.show()
"""),
        md(f"""
Knowing the market's own direction every quarter would have produced a Sharpe of
{R['ceiling']:.2f}. A perfect forecast of the coming quarter's economy captures about
{R['head_cap']} of that gap — it calls the market's direction about as well as a coin.
"""),
        md(f"""
## Beat 5 · The verdict

**Signal: {R['signal']}.** The claim as stated — a perfect forecast of the coming quarter's
economy tells you where stocks are going — fails: {R['head']:.2f} against {R['bh']:.2f}, inside
the range luck produces. The value only appears when the oracle sees beyond the quarter you
hold, and even there the combined oracle does not clear both luck tests.

**Tradability: {R['trad']}.** The lagged rule a real investor could run looks good on paper
({R['real']:.2f}) but falls to {R['strict']:.2f} with one more quarter of delay, and it runs on
revised data that nobody actually had at the time.

**Stocks lead the economy? {R['third']}.** Correlation with GDP growth is {R['ll0']} in the same
quarter but {R['ll1']} and {R['ll2']} one and two quarters later.

> **In one sentence:** {R['one_sentence']}
"""),
        md("""
## Beat 6 · Could you trade it?

Not as advertised. The only rule you could run with real information is the lagged one, and
three things undercut it: its edge disappears if the data arrive a quarter later; it was
measured on numbers revised years after the fact (real-time GDP is often off by a percentage
point or more); and its higher Sharpe comes from sitting out volatile quarters, not from extra
return. Costs are not the problem — it trades about once a year.
"""),
        md("""
## Beat 7 · Going further \U0001F6AA

- **Real-time data.** Rerun the h = −1 leg on the Philadelphia Fed's real-time dataset
  (Croushore & Stark) — the as-first-published GDP numbers. Our guess: whatever is left shrinks.
- **The forecasters themselves.** Swap the oracle for the Survey of Professional Forecasters'
  consensus. If perfect foresight of the coming quarter is worth ~nothing, an imperfect one
  should be worth less.
- **Monthly.** Does knowing next month's payrolls perfectly do better than knowing next
  quarter's GDP? Neighbouring studies (payrolls day, GDPNow revisions) suggest the market prices
  even that within minutes.
- **The useful direction.** The market is a *forecaster* of the economy. That is the trade worth
  studying: stocks as a recession indicator, not the economy as a stock indicator.
"""),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(f"""
# The Perfect Macro Forecaster — a quantitative teardown \U0001F52C
### Lead-lag · an oracle at every information horizon · block bootstrap · rotation placebo · the ceiling

{BADGES}

The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We grant
perfect, final-vintage foresight of US macro (real GDP growth, Δunemployment, CPI inflation,
ΔT-bill; `statsmodels` `macrodata`, 1959Q1–2009Q3) and measure what it is worth for timing the
US market (Fama-French `Mkt-RF + RF`, **total return**, monthly compounded to quarters). The
decisive comparisons are **§3** (the oracle grid with inference) and **§4** (same quarter vs one
quarter beyond).

> ⚠️ **Not investment advice.** Headline numbers mirror the pinned run in
> [`docs/results.md`](../docs/results.md) (as-of {R['as_of']}, fingerprint
> `{R['fingerprint']}`); the cells below recompute them on the same bundled tapes (bootstrap
> draws reduced for speed, so third-decimal p-values may differ).
>
> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back into intuition.
"""),
        code(BOOT),
        md(f"""
## Verdict, up front

| Axis | Stamp | Decisive numbers |
|---|---|---|
| **Signal** — is perfect coming-quarter macro foresight worth something? | {R['signal']} | Combined oracle h = 0: Sharpe {R['head']:.2f} vs B&H {R['bh']:.2f}, ΔSharpe {R['head_d']}, bootstrap p = {R['head_boot']:.2f}, rotation p = {R['head_rot']:.2f}. At h = +1: {R['lead']:.2f} (p = {R['lead_boot']:.2f} / {R['lead_rot']:.2f}) — suggestive, not certified. GDP alone at h = +1: {R['g_lead']:.2f} (p = {R['g_lead_boot']:.2f} / {R['g_lead_rot']:.3f}). |
| **Tradability** — what survives the release lag? | {R['trad']} | h = −1: {R['real']:.2f} (ΔSharpe {R['real_d']}, bootstrap p = {R['real_boot']:.2f}); h = −2: {R['strict']:.2f} (ΔSharpe {R['strict_d']}). Final vintage caps the stamp at Fragile. |
| **Stocks lead the economy?** | {R['third']} | corr(ex_t, g_t+k): k=0 {R['ll0']} (t {R['ll0_t']}), k=+1 {R['ll1']} (t {R['ll1_t']}), k=+2 {R['ll2']} (t {R['ll2_t']}). |

Ceiling (perfect market oracle): **{R['ceiling']:.2f}**. Window {R['window']}, {R['n_q']} quarters,
identical for every rule.

> \U0001F4A1 **In plain words.** A perfect economic forecast for the coming quarter is already in
> the price. You would need to see further than the market does — and nobody does.
"""),
        md("""
## 1 · Hypotheses (fixed before the run)

- **H₁ (the claim).** The combined oracle at h = 0 beats buy-and-hold net of 10 bps, with
  bootstrap p < 0.05 **and** rotation p < 0.05 → Signal Real. If not, but h = +1 does → Mixed.
- **H₂ (markets price the future).** The excess return correlates with GDP growth one or two
  quarters *later* (NW *t* ≥ 2) more than with the same quarter.
- **H₃ (what is knowable).** The combined oracle at h = −1 beats buy-and-hold with bootstrap
  p < 0.10 → Fragile, else Mirage. Never Investable on final-vintage data.

**Execution lag.** One: the position for quarter *t* is fixed at the close of *t − 1*. All
look-ahead lives in the oracle's *h*. Costs one-way × |Δposition| × NAV.
"""),
        md("""
## 2 · Lead-lag: who moves first
"""),
        code("""
rows = {}
for v in data.MACRO_VARS:
    L = st.lead_lag(panel, v)
    rows[data.LABELS[v]] = [f"{c:+.2f} ({t:+.1f})" for c, t in zip(L["corr"], L["t"])]
LL = pd.DataFrame(rows, index=[f"k={k:+d}" for k in range(-4, 5)]).T
print("corr(excess return_t, x_{t+k})  (Newey-West t, 4 lags)")
print(LL.to_string())

fig, ax = plt.subplots(figsize=(10, 4.8))
for v, c in zip(data.MACRO_VARS, (BLUE, BAD, AMBER, GREY)):
    L = st.lead_lag(panel, v)
    ax.plot(L.index, L["corr"], "o-", lw=2, color=c, label=data.LABELS[v])
ax.axhline(0, color="k", lw=1); ax.axvline(0, color=GREY, lw=1)
ax.set_xlabel("k (macro print k quarters after the return quarter)")
ax.set_ylabel("correlation"); ax.legend(fontsize=9)
plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** GDP and unemployment line up with *future* stock returns'
> mirror image: this quarter's market move is about the economy six months from now. Inflation
> barely lines up with anything at the quarterly frequency.

Checking that the HAC *t* is honestly sized when the regressor is autocorrelated:
"""),
        code("""
print(st.lead_lag_size(n_seeds=200))
"""),
        md("""
## 3 · The oracle grid, with inference
"""),
        code("""
T, win = st.horizon_table(panel, n_boot=1000)
sub = panel.loc[win]
bh = st.ann_sharpe(sub["ex"])
ceil = st.ann_sharpe(st.book(st.market_oracle(sub), sub["ex"]))
print(f"window {win[0].date()} -> {win[-1].date()} ({len(win)} quarters); B&H {bh:.2f}; ceiling {ceil:.2f}")
show = T[["var", "h", "sharpe", "dsharpe", "ci_lo", "ci_hi", "boot_p", "rot_p", "active_t",
          "exposure", "switches_per_year", "hit_rate", "capture"]]
print(show.round(3).to_string(index=False))
print(f"\\ncells clearing both nulls at 5%: {sum(st.passes(r) for r in T.to_dict('records'))} of {len(T)}")
"""),
        code("""
piv = T.pivot(index="var", columns="h", values="dsharpe").loc[["g", "du", "infl", "dtb", "combined"]]
fig, ax = plt.subplots(figsize=(9, 4.6))
lim = np.abs(piv.to_numpy()).max()
im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"h={h:+d}" for h in piv.columns])
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([data.LABELS.get(v, v) for v in piv.index])
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="ΔSharpe vs buy-and-hold (net)")
ax.set_title("value of perfect foresight, by variable and horizon", fontsize=10)
ax.grid(False)
plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** Blue cells beat holding the market. The h = 0 column — a
> perfect forecast of the quarter you hold — is pale. The colour lives to the right (seeing
> beyond the quarter) and, for the combined vote, oddly at h = −1, which §5 takes apart.

### 3.1 The rotation placebo, drawn
"""),
        code("""
P = st.all_positions(panel).loc[win]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, name in zip(axes, ("combined@+0", "combined@+1")):
    rt = st.rotation_test(P[name], sub["ex"])
    ax.hist(rt["null"], bins=30, color=GREY)
    ax.axvline(rt["sharpe"], color=BAD if rt["p"] > 0.05 else GOOD, lw=2.5,
               label=f"actual {rt['sharpe']:.2f}, p = {rt['p']:.3f}")
    ax.axvline(bh, color="k", ls="--", lw=1.2, label=f"buy and hold {bh:.2f}")
    ax.set_title(f"{name}: {rt['n_rot']} rotations", fontsize=10)
    ax.set_xlabel("Sharpe of the rotated oracle"); ax.legend(fontsize=8.5)
plt.show()
"""),
        md("""
## 4 · Same quarter vs one quarter beyond — the market prices the future
"""),
        code("""
for v in ("g", "du", "combined"):
    a = st.book(P[f"{v}@+1"], sub["ex"]); b = st.book(P[f"{v}@+0"], sub["ex"])
    bt = st.sharpe_diff_bootstrap(a.to_numpy(), b.to_numpy(), n_boot=1000)
    print(f"{v:9s} h=+1 {st.ann_sharpe(a):.2f} vs h=0 {st.ann_sharpe(b):.2f}   "
          f"diff {bt['dsharpe']:+.2f}  95% CI [{bt['ci_lo']:+.2f}, {bt['ci_hi']:+.2f}]  p={bt['p']:.3f}")
"""),
        md(f"""
For GDP the step from h = 0 to h = +1 is worth {R['contrast_g']} of Sharpe (bootstrap
p = {R['contrast_g_p']:.3f}). This is the study's mechanism: the quarter-ahead forecast the
industry produces is exactly the horizon the market has already discounted.

> \U0001F4A1 **In plain words.** Knowing *today's* weather perfectly doesn't help you bet on
> umbrellas — the umbrella shops already stocked up last week.
"""),
        md("""
## 5 · What a real investor could know — the release lag
"""),
        code("""
r = T[(T["var"] == "combined") & (T["h"].isin([-2, -1]))][["h", "sharpe", "dsharpe", "boot_p", "rot_p", "active_t"]]
print(r.round(3).to_string(index=False))
print()
print(st.cost_sweep(P["combined@-1"], sub).round(3).to_string())
print()
print(st.subperiods(P["combined@-1"], sub).round(3).to_string())
"""),
        md(f"""
Three red flags on the lagged rule: (1) one more quarter of lag turns {R['real']:.2f} into
{R['strict']:.2f}; (2) the Newey-West *t* of the active return is ≈ 0, so the Sharpe gain is a
volatility-avoidance effect, not extra return; (3) it is one of {2 * 5} realistic cells, on
**final-vintage** data. Revisions to GDP growth routinely exceed a percentage point (annualised),
large enough to flip an above/below-median call. Hence **Fragile**, capped by construction.

> \U0001F4A1 **In plain words.** The version you could actually run looks okay only if you
> squint, use numbers nobody had at the time, and get the timing exactly right.
"""),
        md("""
## 6 · The ceiling
"""),
        code("""
lad = st.ceiling_ladder(panel, win)
print(lad.round(3).to_string())
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(lad["hit_rate"] * 100, lad["sharpe"], "o-", color=GREY, lw=2, label="market oracle, corrupted")
for _, r in T.iterrows():
    c = GOOD if r["h"] > 0 else (BLUE if r["h"] == 0 else AMBER)
    ax.scatter(r["hit_rate"] * 100, r["sharpe"], color=c, s=28, alpha=0.8)
ax.scatter([], [], color=GOOD, label="macro oracles, h > 0")
ax.scatter([], [], color=BLUE, label="macro oracles, h = 0")
ax.scatter([], [], color=AMBER, label="macro oracles, h < 0")
ax.axhline(bh, color="k", ls="--", lw=1.2, label="buy and hold")
ax.set_xlabel("hit rate on the market's direction (%)"); ax.set_ylabel("Sharpe (net)")
ax.legend(fontsize=8.5)
plt.show()
"""),
        md("""
> \U0001F4A1 **In plain words.** All 25 macro oracles sit in a cloud between 45% and 61% hit
> rates — the "barely better than a coin" region — far from the 100% corner.
"""),
        md("""
## 7 · Synthetic control — the machinery, not the market
"""),
        code("""
sp = st.synthetic_power(seeds=range(1014, 1020), n_boot=300)
print(sp.groupby("signal_strength")[["dsharpe", "boot_p", "rot_p", "passes", "lead_t"]].mean().round(3).to_string())
"""),
        md("""
With the lead planted (strength 1) the GDP h = +1 oracle clears both nulls in every seed; on the
matched null (strength 0) it does not. A harness that can bank a planted effect and refuses a
null one is the precondition for reading its "no" on the real tape as a real "no".
"""),
        md("""
## 8 · Caveats

- **Final vintage.** `macrodata` is revised data. T-bills are never revised, CPI barely,
  unemployment only via seasonal factors; GDP heavily. Every h < 0 number is an upper bound.
- **Generous release lag.** h = −1 assumes last quarter's GDP is known at quarter start; the
  advance estimate arrives ~4 weeks later.
- **One rule family.** Median split, all-in/all-out. Graded rules could extract more; the ceiling
  bounds any of them.
- **Multiplicity.** 25 cells; the verdict uses three pre-registered ones.
- **Sample.** 1964–2009, US, ~8 recessions.
"""),
        md(f"""
## 9 · Verdict, could you trade it, going further

**Signal: {R['signal']}** · **Tradability: {R['trad']}** · **Stocks lead the economy? {R['third']}.**

> **In one sentence:** {R['one_sentence']}

**Could you trade it?** No version that uses information you could have had beats the market
robustly; costs are irrelevant at ~1 switch a year; capacity is irrelevant for an index timer.
The binding constraints are the information set and the revisions.

**Going further.**
- Rerun h = −1 on the Philadelphia Fed real-time dataset (first-release GDP).
- Replace the oracle with the Survey of Professional Forecasters consensus.
- Graded rules: regress the excess return on the oracle value with an expanding window.
- Invert the question: the market as a forecaster of GDP (Fama 1990's direction).
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
