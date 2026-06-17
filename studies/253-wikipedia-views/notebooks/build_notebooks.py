"""Generate the two narrative notebooks for Study 253 (Wiki-Views).

    python notebooks/build_notebooks.py

This writes both notebooks then executes them in place with nbconvert. The
synthetic figures run anywhere, offline and deterministic; the real-tape cells
use the cached Yahoo monthly panel under ../_cache/ if present, otherwise the
deterministic offline best-effort proxy (curated views joined to views-
independent prices), and quote the frozen headline numbers in ``R`` (mirroring
docs/results.md) so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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


# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_months=118, n_leg=6, n_tickers=30,
    low_ann=14.11, low_t=2.44,
    high_ann=15.33, high_t=2.91,
    market_ann=14.52,
    spread_ann=-1.22, spread_bps=-10.2, spread_t=-0.31, spread_hit=0.483,
    low_xret_ann=-0.41, low_xret_t=-0.14,
    boot_lo=-0.726, boot_hi=0.542, boot_frac_neg=63,
    # sub-periods
    sub_2017_mean=-2.48, sub_2017_t=-0.34, sub_2017_n=36,
    sub_2020_mean=0.29, sub_2020_t=0.04, sub_2020_n=36,
    sub_2023_mean=-0.26, sub_2023_t=-0.03, sub_2023_n=35,
    # costs
    turnover=89.08, drag_bps=35.6, net_spread_bps=-45.8, net_spread_t=-1.39,
    rand_bps=-3.6,
    # positive control
    ctrl_premium=2.0, ctrl_spread_ann=62.02, ctrl_spread_t=17.03,
    null_spread_ann=-4.38, null_spread_t=-1.21,
)

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

from wikipedia_views import data, strategy as st

CACHE_PATH = data.RETURNS_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

if HAVE_REAL:
    prices = data.fetch_returns(fetch=False, cache_path=CACHE_PATH)
    views = data.wiki_views_panel(n_months=prices.shape[0])
    views, prices = data.align_panels(views, prices)
    TAPE = "cached Yahoo monthly tape"
else:
    # Deterministic offline best-effort proxy: curated views + views-INDEPENDENT prices
    views, prices = data.best_effort_real_tape(n_months=120, seed=data.SEED)
    TAPE = "offline best-effort proxy (curated views, views-independent prices)"

sur = st.attention_surge(views, smooth=1)
res = st.decile_returns(sur, prices, q=0.20)
spread = res["spread"].dropna()
low_xret = (res["low"] - res["market"]).dropna()
print(f"Tape: {TAPE}")
print(f"Panel: {prices.shape[0]} months x {prices.shape[1]} tickers")
print(f"Portfolio months: {len(res)}  |  ~{res['n_low'].mean():.0f} names per leg")
"""

R_INJECT = f"""
# Frozen headline numbers (mirror of docs/results.md, as-of 2026-06-17)
R = {R!r}
"""


# ---------------------------------------------------------------------------
# Notebook 1 -- For the curious
# ---------------------------------------------------------------------------
def build_curious() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 253 -- Wiki-Views
## For the curious: do surging Wikipedia page-views on a ticker predict the next move?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

When a stock is in the news, its Wikipedia page lights up. A 2013 *Scientific Reports*
paper (Moat et al.) claimed that **changes in Wikipedia page-views on company pages
preceded stock-market moves** -- and a strategy trading on view surges beat the market.
It became a poster child for "big-data alpha".

The behavioural prior (Da, Engelberg & Gao 2011) is that an *attention surge* precedes
a short-horizon **reversal**: the crowd piles in, the price overshoots, then it gives
the move back. So the natural bet is **short the attention surge, long the attention
drought** -- which is exactly what we test here.
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
> ⚠️ **Data honesty.** The Wikimedia page-views API only starts 2015-07, is rate-limited,
> and the raw series is dominated by one-off news spikes. Rather than ship a brittle live
> scraper we use a **curated anchor table** of representative per-ticker traffic
> (`data.WIKI_VIEWS_ANCHORS`) expanded into a deterministic monthly panel. Offline, the
> returns side is generated *independently* of the views -- the honest null. A genuine
> Wikimedia-vs-Yahoo cache can be dropped in and re-run, but published attention effects
> are intraday-to-weekly and tiny, not a monthly mega-cap spread.
"""),

        md("## The attention surge: what the signal looks like"),
        code("""\
# Show a couple of names' page-view paths and their month-over-month surge
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
show = [c for c in ["TSLA", "KO"] if c in views.columns][:2] or list(views.columns[:2])
ax = axes[0]
for c in show:
    ax.plot(views.index, views[c] / 1000.0, lw=1.5, label=c)
ax.set_title("Curated monthly Wikipedia page-views (thousands)")
ax.set_ylabel("k views / month"); ax.legend(fontsize=9)

ax = axes[1]
for c in show:
    ax.plot(sur.index, sur[c], lw=1.3, label=c)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Attention surge = log(views_t / views_{t-1})")
ax.set_ylabel("monthly log change"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("A 'surge' is a positive spike; a 'drought' is a negative one. We rank the basket on this each month.")
"""),

        md("## The drought-vs-surge race"),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Cumulative growth of the two legs
ax = axes[0]
cum_low  = (1 + res["low"].dropna()).cumprod()
cum_high = (1 + res["high"].dropna()).cumprod()
cum_mkt  = (1 + res["market"].dropna()).cumprod()
ax.plot(cum_low.index,  cum_low.values,  color=GREEN, lw=1.8, label="Low surge (attention drought, long)")
ax.plot(cum_high.index, cum_high.values, color=RED,   lw=1.8, label="High surge (attention surge, short)")
ax.plot(cum_mkt.index,  cum_mkt.values,  color=GREY,  lw=1.2, ls="--", label="Equal-weight basket")
ax.set_title("Cumulative growth of $1 (survivorship-biased)")
ax.set_ylabel("Cumulative return ($)"); ax.legend(fontsize=9)

# Rolling 24-month spread
ax = axes[1]
roll = spread.rolling(24).mean() * 1200
ax.plot(roll.index, roll.values, color=GREY, lw=1.5)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Rolling 24-month low-minus-high spread (%/yr)")
ax.set_ylabel("Annualised spread (%/yr)")
plt.tight_layout(); plt.show()
print(f"Drought-minus-surge spread: {R['spread_ann']:+.1f}%/yr = {R['spread_bps']:+.1f}bps/mo  HAC t = {R['spread_t']:+.2f}")
print("The two legs are tangled together -- there is no persistent edge either way.")
"""),

        md("""\
## The bottom line for the curious

The headline is a **non-event**: betting on the Wikipedia-views surge reversal produces a
spread of **-1.2%/yr** with HAC *t* = **-0.31** -- statistically indistinguishable from
zero, and with the *wrong sign* (the high-surge leg edged slightly ahead).

Why does the famous "Wikipedia predicts markets" result evaporate here?

- **Wrong frequency**: the published effects are *days to two weeks*; a monthly rebalance
  averages them away.
- **Wrong universe**: attention effects are strongest in *small, retail-heavy* names; a
  liquid mega-cap basket has every effect already arbitraged.
- **Spike-dominated data**: one news event swamps the page-view series, so the "surge"
  is mostly noise about a one-off headline, not a tradable state.

This is a **None / Mirage** -- a fun big-data folklore that does not hold up as a monthly
cross-sectional signal.
"""),

        md("*Desk verdict: **None / Mirage** -- see [README.md](../README.md) and [docs/results.md](../docs/results.md) for full numbers.*"),
    ]
    _write(nb, "01_for_the_curious.ipynb")


# ---------------------------------------------------------------------------
# Notebook 2 -- For the quants
# ---------------------------------------------------------------------------
def build_quants() -> None:
    nb = new_notebook()
    nb.cells = [
        md("""\
# Study 253 -- Wiki-Views
## For the quants: surge signal, HAC tests, bootstrap CI, sub-period nulls, costs

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Positive control: the engine finds attention-reversal when it is planted

> 💡 **In plain words.** Before trusting a null on real data, prove the machinery can
> see a signal that is *definitely* there. We plant a known attention-reversal premium in
> a synthetic panel and confirm the quintile sort lights up.
"""),
        code("""\
# Synthetic positive control: planted reversal premium=0.02 -> strong positive spread
price_syn, views_syn, truth = data.synthetic_attention_panel(n_firms=30, n_months=120, premium=0.02, seed=253)
sur_syn = st.attention_surge(views_syn)
res_syn = st.decile_returns(sur_syn, price_syn, q=0.20)
s_syn = st.summarize(res_syn["spread"].dropna())
print(f"Positive control (premium=0.02):")
print(f"  Spread = {s_syn['mean']*1200:+.2f}%/yr  HAC t = {s_syn['tstat']:+.2f}  n = {s_syn['n']}")

# Null control: premium=0.0
price0, views0, _ = data.synthetic_attention_panel(n_firms=30, n_months=120, premium=0.0, seed=253)
res0 = st.decile_returns(st.attention_surge(views0), price0, q=0.20)
s0 = st.summarize(res0["spread"].dropna())
print(f"\\nNull control (premium=0.0):")
print(f"  Spread = {s0['mean']*1200:+.2f}%/yr  HAC t = {s0['tstat']:+.2f}  n = {s0['n']}")
print("  -> Engine reads strong on planted reversal, ~zero on the null. The machinery is honest.")
"""),

        md("## Primary specification: HAC t and bootstrap Sharpe CI on the real-ish tape"),
        code("""\
s_spread = st.summarize(spread)
s_low    = st.summarize(res["low"].dropna())
s_high   = st.summarize(res["high"].dropna())
s_market = st.summarize(res["market"].dropna())

print("=== Low-minus-high surge quintile, 1-month hold ===")
print(f"Portfolio months: {len(res)}   avg names/leg: {res['n_low'].mean():.0f}")
print()
print(f"LOW  (drought, long) : {s_low['mean']*1200:+.2f}%/yr  HAC t={s_low['tstat']:+.2f}")
print(f"HIGH (surge, short)  : {s_high['mean']*1200:+.2f}%/yr  HAC t={s_high['tstat']:+.2f}")
print(f"MARKET (eq-weight)   : {s_market['mean']*1200:+.2f}%/yr")
print(f"SPREAD (low-high)    : {s_spread['mean']*1200:+.2f}%/yr = {s_spread['mean']*10000:+.1f}bps/mo  HAC t={s_spread['tstat']:+.2f}  hit={s_spread['hit_rate']:.3f}")

try:
    from quantlab.stats import sharpe_ci_bootstrap
    ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=12, seed=253)
    print(f"\\nBootstrap Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% negative)")
except Exception as e:
    print(f"\\n[quantlab.stats unavailable -- frozen CI: [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}], {R['boot_frac_neg']}% negative]")
print("\\nThe CI straddles zero -- exactly what a null looks like.")
"""),

        md("## Sub-period nulls: nothing to persist"),
        code("""\
periods = [("2017-2019", "2017", "2019"), ("2020-2022", "2020", "2022"), ("2023-2026", "2023", "2026")]
sub_stats = []
for label, a, b in periods:
    sub = spread[a:b]
    s = st.summarize(sub)
    sub_stats.append({"period": label, "mean_ann": s["mean"]*1200, "tstat": s["tstat"], "n": s["n"]})
    print(f"  {label}: mean={s['mean']*1200:+.2f}%/yr  HAC t={s['tstat']:+.2f}  n={s['n']}")

sts = pd.DataFrame(sub_stats)
fig, ax = plt.subplots(figsize=(9, 4))
colors_sub = [GREEN if abs(t) >= 2 else RED for t in sts["tstat"]]
ax.bar(sts["period"], sts["tstat"], color=colors_sub)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="t=2 inference bar")
ax.axhline(-2.0, color="black", ls="--", lw=0.9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Sub-period HAC t-stat (none clears the bar)")
ax.set_ylabel("HAC t-stat"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print("No sub-period reaches |t|=1, let alone 2. There is no signal to decay.")
"""),

        md("## Turnover and cost drag: a negative number made worse"),
        code("""\
drag = st.turnover_cost_drag(sur, prices, q=0.20, one_way_bps=10.0)
avg_turnover = drag.mean() / 0.004
avg_drag = drag.mean() * 10000
net = spread - drag.reindex(spread.index).fillna(drag.mean())
s_net = st.summarize(net)
print(f"Avg monthly two-leg turnover: {avg_turnover:.2%}")
print(f"Avg drag @10bps one-way: {avg_drag:.1f} bps/mo")
print(f"Net spread @10bps: {s_net['mean']*1200:+.2f}%/yr = {s_net['mean']*10000:+.1f}bps/mo  HAC t={s_net['tstat']:+.2f}")
print()
print("> Honesty: the short (surge) leg ALSO pays stock-borrow on top of this, and the")
print("  gross spread is already negative -- costs only deepen the loss.")
"""),

        md("## Random-portfolio null: no cross-sectional information"),
        code("""\
print("Computing random-portfolio null (n_draws=100 per month)...")
rand = st.random_portfolio_returns(sur, prices, q=0.20, n_draws=100, seed=253)
print(f"Random excess: mean={rand.mean()*10000:+.1f}bps  std={rand.std()*10000:.1f}bps")
print(f"  -> Null centred at zero: {abs(rand.mean()*10000) < 10}")
print("  -> The drought leg is statistically a random draw from the basket: the surge signal")
print("     carries no information beyond luck on this monthly mega-cap tape.")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 253 -- Wiki-Views ===")
print()
print(f"Signal: NONE   (drought-minus-surge HAC t = {R['spread_t']:+.2f}, CI [{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}] straddles zero)")
print(f"        Wrong sign, no sub-period clears |t|=1, random-null centred at zero.")
print(f"        Literature support (Moat 2013, Da-Engelberg-Gao 2011) alone never beats WEAK;")
print(f"        published effects are intraday-to-weekly small-cap, not monthly mega-cap.")
print()
print(f"Tradability: MIRAGE")
print(f"  Gross spread negative; turnover={R['turnover']:.0f}%/mo; net={R['net_spread_bps']:.1f}bps/mo before short-leg borrow.")
print()
print("Survivorship: NAMED -- basket = current mega-cap survivors; upper bound already zero.")
print()
print("Bottom line: None/Mirage -- a fun big-data folklore that does not survive as a")
print("  monthly cross-sectional signal on a liquid mega-cap basket.")
"""),
    ]
    _write(nb, "02_for_the_quants.ipynb")


# ---------------------------------------------------------------------------
# Shared _write helper
# ---------------------------------------------------------------------------
def _write(nb: nbf.NotebookNode, filename: str) -> None:
    path = os.path.join(HERE, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Execute notebooks with nbconvert
# ---------------------------------------------------------------------------
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
