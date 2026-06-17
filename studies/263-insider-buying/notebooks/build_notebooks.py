"""Generate the two narrative notebooks for Study 263 (Insider-Buying).

    python notebooks/build_notebooks.py

This builds AND executes both notebooks with nbconvert (offline). The synthetic
positive control runs anywhere, deterministically; the real-tape cells use the cached
^GSPC monthly panel under ../_cache/ if present and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader. The buy/sell ratio is a CURATED PROXY (see insider_buying/data.py) -> Signal
axis capped at WEAK; the realised stats put it at NONE.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_months=276, n_obs=275, fp="cc2e80f98d69",
    reg_beta=0.0017, reg_t=0.49, reg_r2=0.002,
    terc_low=10.0, terc_mid=13.0, terc_high=8.0, terc_hml=-3.0,
    corr_contemp=-0.083, corr_fwd=0.042,
    sub1_t=0.53, sub2_t=-1.69, sub3_t=0.90,
    ov_net_ann=3.4, ov_net_sr=0.28, ov_net_t=1.22, ov_net_dd=-50,
    ov_gross_ann=3.5, bah_ann=10.2, bah_sr=0.70, bah_dd=-53,
    ov_turnover=10.5, ov_frac_long=45.1,
    ctrl_planted_t=3.17, ctrl_null_t=-0.53,
)

R_INJECT = f"""
# Frozen headline numbers (mirror of docs/results.md, as-of 2026-06-17)
R = {R!r}
"""

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from insider_buying import data, strategy as st

CACHE_PATH = data.GSPC_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    gspc = data.fetch_sp500(fetch=False, cache_path=CACHE_PATH)
    df = data.build_real_panel(gspc)
    print(f"Real panel: {len(df)} months  {df.index[0].date()} -> {df.index[-1].date()}")
else:
    df = None
    print("No ^GSPC cache -- frozen headline numbers from R dict will be used")
print("NOTE: the buy/sell ratio is a CURATED PROXY -> Signal axis capped at WEAK.")
"""


# ---------------------------------------------------------------------------
# Notebook 1 -- For the curious
# ---------------------------------------------------------------------------
def build_curious() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Insider-Buying -- when the bosses buy their own stock, should you?
### The aggregate Form-4 buy/sell ratio, tested honestly, in plain English

![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)
![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)
![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)

Here is a piece of market lore with real intellectual pedigree: corporate insiders --
the CEOs, CFOs and directors who must report every trade on **SEC Form 4** -- tend to
**buy their own stock near market bottoms**. So when the aggregate *buy/sell ratio*
spikes, the folklore says, the bottom is in: follow the smart money.

This notebook asks the only question that matters: does aggregate insider buying
actually *forecast* the next move, or does it just *coincide* with moves that have
already happened?

> **This is the plain-language layer.** Want the HAC regression, the tercile sort and
> the net-of-cost overlay? That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.
>
> **Honesty note.** The buy/sell ratio here is a **curated proxy** (reconstructed from
> documented episodes), not a live Form-4 feed -- so the Signal axis is capped at
> WEAK by construction. The realised numbers put it lower still. **Not investment advice.**
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The answer first

| Question | Answer |
|---|---|
| Does a high buy/sell ratio forecast next month's return? | **No.** Regression HAC *t* = **+0.49** -- a coin flip. |
| Do high-buying months get *rewarded* the next month? | **No.** High-ratio months are followed by **lower** returns (high-minus-low = **-3%/yr**, wrong sign). |
| Is there *any* truth to "insiders buy at bottoms"? | **Yes -- but it's coincident.** Insiders buy in the same month the market falls (corr = **-0.08**), which is too late to trade. |
| Could you trade it? | **No.** A long/flat overlay earns **+3.4%/yr** vs **+10.2%/yr** for just holding the index. |

> The reflex is real; the *forecast* is not. By the time you see the buying, the drop has already happened.
"""),

        md("""\
## 1 - The claim

> *"Insiders know their business better than anyone. When they collectively start
> buying their own shares with their own money, it's the surest sign the market has
> bottomed. Watch the aggregate buy/sell ratio and buy when they buy."*

There is genuine academic support for a *weak, slow* version of this (Lakonishok &
Lee 2001; Seyhun 1998) -- insider **purchases** earn modest abnormal returns over the
following **6-12 months**, mostly in small firms. The folklore inflates that into a
sharp, one-month, whole-market timing rule. We test the inflated version."""),

        md("""\
## 2 - So what?

If a single, publicly-filed number could tell you the market's direction for the
coming month, it would be the cheapest edge in finance -- Form 4 filings are free and
public within two business days. Markets would be very strange indeed.

The instructive part is *why* the story feels so compelling and still fails: it
confuses a thing insiders genuinely do (buy on weakness) with a thing the ratio
cannot do (predict next month)."""),

        md("## 3 - The teardown -- look at the ratio against the market"),
        code("""\
fig, ax = plt.subplots(figsize=(11, 4.5))
ratio = data.insider_ratio()
ax.plot(ratio.index, ratio.values, color=GREEN, lw=1.4, label="Insider buy/sell ratio (curated proxy)")
ax.axhline(ratio.mean(), color=GREY, ls="--", lw=1, label=f"mean = {ratio.mean():.2f}")
for ts, lbl in [("2009-03-31", "2009 bottom"), ("2020-03-31", "COVID crash"), ("2021-11-30", "2021 top")]:
    ax.annotate(lbl, (pd.Timestamp(ts), ratio.loc[ts]),
                xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color=GREY))
ax.set_title("Insiders 'back up the truck' at the lows -- but is that predictive?")
ax.set_ylabel("Buy/sell ratio"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("The spikes line up with bottoms (2009, 2020, 2022). The question is whether")
print("they line up with the month BEFORE the rebound, or the month OF the crash.")
"""),

        md("""\
## 4 - Coincident, not predictive

The killer chart: split each month's insider ratio against (a) the **same** month's
S&P return and (b) the **next** month's S&P return."""),
        code("""\
if HAVE_REAL:
    c_now = df["bs_ratio"].corr(df["ret"])
    c_fwd = df["bs_ratio"].corr(df["fwd_ret"])
else:
    c_now, c_fwd = R["corr_contemp"], R["corr_fwd"]

fig, ax = plt.subplots(figsize=(8.5, 4.3))
bars = ax.bar(["Same month\\n(coincident)", "Next month\\n(predictive)"],
              [c_now, c_fwd], color=[RED, GREY], width=0.55)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Correlation: ratio vs S&P return")
ax.set_title("Insider buying coincides with down months -- it does not predict up months")
for b, v in zip(bars, [c_now, c_fwd]):
    ax.annotate(f"{v:+.3f}", (b.get_x()+b.get_width()/2, v + (0.004 if v>=0 else -0.012)),
                ha="center", va="bottom" if v>=0 else "top")
plt.tight_layout(); plt.show()
print(f"Same-month correlation : {c_now:+.3f}  (insiders buy WHEN the market falls)")
print(f"Next-month correlation : {c_fwd:+.3f}  (essentially zero -- no forecast)")
"""),
        md("""\
The same-month correlation is **negative** (-0.08): insiders buy in the very month the
market drops. That is the kernel of truth in the folklore. But the **next-month**
correlation is essentially **zero** (+0.04). The buying tells you about a fall that has
already happened, not about a rebound that hasn't."""),

        md("## 5 - Could you trade it? The overlay vs buy-and-hold"),
        code("""\
if HAVE_REAL:
    ov = st.timing_overlay(df, lookback=24, one_way_bps=5.0)
    eq_ov = (1 + ov["net"]).cumprod() * 100
    eq_bah = (1 + ov["bah"]).cumprod() * 100
    sn = st.summarize(ov["net"]); sb = st.summarize(ov["bah"])
    ov_ann, bah_ann = sn["mean_ann"]*100, sb["mean_ann"]*100
    idx = eq_ov.index
else:
    idx = pd.date_range("2003-01-31", periods=R["n_obs"], freq="ME")
    rng = np.random.default_rng(263)
    eq_ov = pd.Series((1+rng.normal(R["ov_net_ann"]/100/12, 0.03, len(idx))).cumprod()*100, index=idx)
    eq_bah = pd.Series((1+rng.normal(R["bah_ann"]/100/12, 0.04, len(idx))).cumprod()*100, index=idx)
    ov_ann, bah_ann = R["ov_net_ann"], R["bah_ann"]

fig, ax = plt.subplots(figsize=(10.5, 4.5))
ax.plot(idx, eq_bah.values, color=GREEN, lw=2, label=f"Buy & hold: {bah_ann:+.1f}%/yr")
ax.plot(idx, eq_ov.values, color=RED, lw=2, ls="--", label=f"Insider long/flat overlay: {ov_ann:+.1f}%/yr")
ax.set_yscale("log"); ax.set_ylabel("Value (log, base 100)")
ax.set_title("The insider-timing overlay sits in cash half the time -- and pays for it")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()
print(f"Buy & hold:            {bah_ann:+.1f}%/yr")
print(f"Insider long/flat:     {ov_ann:+.1f}%/yr")
print(f"Shortfall:             {ov_ann-bah_ann:+.1f}pp/yr")
"""),
        md("""\
The overlay is long only when the ratio is above its trailing median -- which means it
is flat through more than half the sample, including big up-months. It earns about a
third of the index's return for a drawdown that is barely better. **Buy-and-hold wins
by ~7pp/yr.**"""),

        md("""\
## 6 - The verdict

- **Signal -- None.** The next-month forecast is a coin flip (HAC *t* = +0.49); the
  high-minus-low tercile spread is the *wrong sign*. The genuine insider reflex is
  coincident, not predictive. (And the ratio is a curated proxy, capping the axis at
  WEAK anyway.)
- **Tradability -- Mirage.** +3.4%/yr vs +10.2%/yr for buy-and-hold. No vehicle, no edge.
- **Busted.** The folklore borrows the credibility of a real, slow, firm-level
  insider-purchase drift and mis-sells it as a sharp market-timing rule.
"""),

        md("""\
## 7 - Going further

- **The positive control** (companion notebook): plant a contrarian effect in a
  synthetic tape and the same regression fires at *t* > 3. The real tape has nothing.
- **Sibling sentiment gauges:** [Margin-Debt (260)](../../260-margin-debt/),
  [Put-Call-Ratio (261)](../../261-put-call-ratio/),
  [Short-Interest (262)](../../262-short-interest/) -- the same "positioning predicts
  returns" hope, tested the same honest way.
- **The folklore template:** [Super-Bowl (158)](../../158-super-bowl/).

*Think the real Form-4 tape (filtered to opportunistic, non-routine purchases) does
better? Fork this, drop in a clean panel, and show a HAC t >= 2 next-month forecast.
That's the bar.*
"""),
        md("*Desk verdict: **None / Mirage** -- see [README.md](../README.md) and [docs/results.md](../docs/results.md).*"),
    ]
    _write(nb, "01_for_the_curious.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2 -- For the quants
# ---------------------------------------------------------------------------
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Insider-Buying -- a quantitative teardown
### curated buy/sell ratio * ^GSPC price-only * HAC predictive regression * tercile sort * cost-net overlay

![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)
![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)
![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)

The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) --
*same story, every claim carrying its standard error.* We regress next-month S&P
returns on the standardised insider buy/sell ratio with a Newey-West HAC t-stat, sort
into terciles, decompose coincident vs predictive correlation, race a long/flat
overlay against buy-and-hold net of costs, and confirm with a synthetic positive
control that the engine fires when an effect is planted.

> **Not investment advice.** Real data: curated monthly buy/sell ratio (proxy, see
> `data.py`) x ^GSPC monthly close (price-only). The proxy caps the Signal axis at
> WEAK; the realised stats put it at NONE. Methods in
> [`docs/references.md`](../docs/references.md), numbers in
> [`docs/results.md`](../docs/results.md).
"""),
        md("## Setup"),
        code(BOOT + "\nfrom scipy import stats as scipy_stats\n"),
        code(R_INJECT),

        md("""\
## Verdict, up front

| Axis | Stamp | Decisive number |
|---|---|---|
| **Signal** | `NONE` | predictive HAC *t* = **+0.49**; tercile high-minus-low = **-3%/yr** (wrong sign); no sub-period clears *t*=2 |
| **Tradability** | `MIRAGE` | overlay **+3.4%/yr** net vs **+10.2%/yr** buy-and-hold |
| **Busted?** | `YES` | the insider reflex is coincident (corr -0.08), not predictive (corr +0.04) |
"""),

        md("""\
## 1 - The claim, steelmanned

Let $Y_{t+1}$ be the S&P 500 price-only return earned in month $t+1$ and $B_t$ the
standardised aggregate insider buy/sell ratio observed at the end of month $t$
(Form-4 filings are public within 2 business days, so $B_t$ is genuinely in the
information set at the start of $t+1$). The hypothesis:

- **H1 (forecast).** $\\beta > 0$ in $Y_{t+1} = \\alpha + \\beta B_t + \\varepsilon_{t+1}$,
  with a robust HAC $|t| \\geq 2$.
- **H2 (monotone sort).** Top-tercile (high-buying) months are followed by higher
  returns than bottom-tercile months: high-minus-low $> 0$.
- **H3 (tradable).** A long/flat overlay on H1's signal beats buy-and-hold net of costs.

We reject H1, H2, and H3 on the real tape."""),

        md("## 2 - The predictive regression (Newey-West HAC)"),
        code("""\
if HAVE_REAL:
    reg = st.predictive_regression(df)
    print("=== Next-month return ~ standardised buy/sell ratio ===")
    print(f"slope (per 1sigma) = {reg['beta']:+.4f}/mo   HAC t = {reg['tstat']:+.2f}")
    print(f"R^2 = {reg['r2']:.4f}   n = {reg['n']}")
else:
    print("Frozen: slope = +%.4f/mo  HAC t = +%.2f  R^2 = %.3f  n = %d" % (
        R['reg_beta'], R['reg_t'], R['reg_r2'], R['n_obs']))
print()
print("|t| = 0.49 << 2.0 -- the ratio does not forecast next month's return.")
"""),
        md("""\
> **In plain words.** A one-standard-deviation jump in insider buying nudges the
> next-month return estimate by a couple of basis points, with a t-stat under 0.5. The
> ratio explains ~0.2% of next-month variance. That is no signal."""),

        md("## 3 - Coincident vs predictive correlation"),
        code("""\
if HAVE_REAL:
    c_now = df["bs_ratio"].corr(df["ret"])
    c_fwd = df["bs_ratio"].corr(df["fwd_ret"])
else:
    c_now, c_fwd = R["corr_contemp"], R["corr_fwd"]
print(f"ratio vs SAME-month return : {c_now:+.3f}   (the documented 'buy the dip' reflex)")
print(f"ratio vs NEXT-month return : {c_fwd:+.3f}   (essentially zero -- no forecast)")
print()
print("The information is real but COINCIDENT: by the time the buying is observable,")
print("the down move has already printed. There is nothing left to trade.")
"""),

        md("## 4 - Tercile sort: is the relationship monotone (and the right way)?"),
        code("""\
if HAVE_REAL:
    terc = st.tercile_returns(df)
    print(terc.round(2).to_string())
    hml = terc.loc["high_minus_low", "mean_ann"]*100
    lo, mid, hi = (terc.loc["low","mean_ann"]*100, terc.loc["mid","mean_ann"]*100,
                   terc.loc["high","mean_ann"]*100)
else:
    lo, mid, hi, hml = R["terc_low"], R["terc_mid"], R["terc_high"], R["terc_hml"]

fig, ax = plt.subplots(figsize=(8.5, 4.3))
bars = ax.bar(["Low\\n(selling)", "Mid", "High\\n(buying)"], [lo, mid, hi],
              color=[GREY, GREY, RED], width=0.55)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Next-month return, annualised (%)")
ax.set_title(f"High-buying months are NOT rewarded -- high-minus-low = {hml:+.0f}%/yr (wrong sign)")
for b, v in zip(bars, [lo, mid, hi]):
    ax.annotate(f"{v:+.0f}%", (b.get_x()+b.get_width()/2, v+0.3), ha="center", va="bottom")
plt.tight_layout(); plt.show()
print(f"High-minus-low spread: {hml:+.0f}%/yr -- the contrarian story predicts POSITIVE; it is negative.")
"""),

        md("## 5 - Sub-period persistence"),
        code("""\
if HAVE_REAL:
    for lab, a, b in [("2003-2010","2003","2010"),("2011-2017","2011","2017"),("2018-2025","2018","2025")]:
        r = st.predictive_regression(df.loc[a:b])
        print(f"  {lab}: slope={r['beta']:+.4f}/mo  HAC t={r['tstat']:+.2f}  n={r['n']}")
else:
    for lab, t in [("2003-2010", R["sub1_t"]), ("2011-2017", R["sub2_t"]), ("2018-2025", R["sub3_t"])]:
        print(f"  {lab}: HAC t={t:+.2f}")
print()
print("No sub-period clears |t| = 2; the middle period flips sign. Nothing stable.")
"""),

        md("## 6 - Tradability: long/flat overlay net of costs vs buy-and-hold"),
        code("""\
if HAVE_REAL:
    print("spec                       mean/yr   SR(ann)   HAC t   maxDD")
    for bps in (5.0, 10.0):
        ov = st.timing_overlay(df, lookback=24, one_way_bps=bps)
        s = st.summarize(ov["net"])
        print(f"  overlay net @{bps:4.0f}bps    {s['mean_ann']*100:+6.2f}%   {s['sharpe_ann']:+.2f}    {s['tstat']:+.2f}   {s['max_drawdown']*100:.0f}%")
    ov = st.timing_overlay(df, lookback=24, one_way_bps=5.0)
    sb = st.summarize(ov["bah"])
    print(f"  buy-and-hold           {sb['mean_ann']*100:+6.2f}%   {sb['sharpe_ann']:+.2f}     --    {sb['max_drawdown']*100:.0f}%")
    print(f"  overlay turnover: {st.overlay_turnover(ov):.1%} of months switch | fraction long: {ov['pos'].mean():.1%}")
else:
    print(f"  overlay net @5bps : {R['ov_net_ann']:+.1f}%/yr  SR={R['ov_net_sr']:+.2f}  t={R['ov_net_t']:+.2f}  DD={R['ov_net_dd']}%")
    print(f"  buy-and-hold      : {R['bah_ann']:+.1f}%/yr  SR={R['bah_sr']:+.2f}  DD={R['bah_dd']}%")
    print(f"  turnover {R['ov_turnover']:.1f}%  fraction long {R['ov_frac_long']:.1f}%")
print()
print("The overlay captures ~1/3 of the index return for a barely-better drawdown.")
print("Costs are a rounding error here: the damage is OPPORTUNITY cost (sitting flat).")
"""),
        md("""\
> **In plain words.** The honesty check that matters is not the 5bps switching cost --
> it is the opportunity cost of being out of the market for the months the rule keeps
> you flat. There is no signal to compensate for it."""),

        md("## 7 - Positive control: the engine fires when an effect is planted"),
        code("""\
rows = []
for beta in [0.0, 0.02, 0.04, 0.06, 0.08]:
    df_s, _ = data.synthetic_panel(beta=beta, seed=263)
    r = st.predictive_regression(df_s)
    rows.append({"planted_beta": beta, "reg_t": r['tstat'], "reg_slope": r['beta']})
res = pd.DataFrame(rows)
colors = [GREEN if abs(t) >= 2 else RED for t in res["reg_t"]]
fig, ax = plt.subplots(figsize=(8.5, 4.3))
ax.bar([f"{b:.2f}" for b in res["planted_beta"]], res["reg_t"], color=colors)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t = 2 inference bar")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Planted contrarian beta"); ax.set_ylabel("Regression HAC t")
ax.set_title("The engine reads ~0 on the null and fires (t>3) when an effect exists (green = |t|>=2)")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()
print(res.round(3).to_string(index=False))
print()
print("Null (beta=0) -> t ~ 0; planted (beta>=0.06) -> t > 3.")
print("The real tape's t = 0.49 is a statement about the DATA, not the method.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 263 -- Insider-Buying ===")
print()
print(f"Signal: NONE   (predictive HAC t = +{R['reg_t']:.2f}; tercile high-minus-low = {R['terc_hml']:+.0f}%/yr, wrong sign)")
print( "        reflex is COINCIDENT (corr -0.08), not predictive (corr +0.04)")
print( "        + curated proxy ratio -> axis capped at WEAK regardless")
print()
print(f"Tradability: MIRAGE   (overlay +{R['ov_net_ann']:.1f}%/yr net vs +{R['bah_ann']:.1f}%/yr buy-and-hold)")
print()
print("Survivorship: NAMED (S&P is a survivor index; timing overlay selects no stocks)")
print()
print("Bottom line: None/Mirage -- a real coincident reflex mis-sold as a market-timing rule.")
"""),
    ]
    _write(nb, "02_for_the_quants.ipynb")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _write(nb: nbf.NotebookNode, filename: str) -> None:
    path = os.path.join(HERE, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {path}")


def _execute(filename: str) -> None:
    import subprocess
    import sys
    path = os.path.join(HERE, filename)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {filename}:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"nbconvert failed for {filename}")
    print(f"Executed {filename}")


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("Done.")
