"""Generate the two narrative notebooks for Study 254 (WSB-Mentions).

    python notebooks/build_notebooks.py

Both notebooks follow the desk beats. The synthetic figures run anywhere,
offline and deterministic; the real-tape cells use the cached meme monthly panel
under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_months=36, n_names=13, n_hype=7,
    hype_ann=-25.9, hype_t=-0.85,
    quiet_ann=-22.4, quiet_t=-1.12,
    basket_ann=-23.2,
    spread_ann=-3.5, spread_bps=-29.3, spread_t=-0.14, spread_hit=0.472,
    spread_sr_lo=-1.105, spread_sr_hi=1.193, spread_frac_neg=0.554,
    hype_xret_ann=-2.7, hype_xret_t=-0.20,
    # sub-periods (calendar-year)
    sub_2021_mean=-49.1, sub_2021_t=-1.01, sub_2021_n=12,
    sub_2022_mean=0.8, sub_2022_t=0.04, sub_2022_n=12,
    sub_2023_mean=37.9, sub_2023_t=0.76, sub_2023_n=12,
    # reversal sign ("short the loud names")
    reversal_ann=3.5, reversal_t=0.14,
    # pooled rank-IC (Spearman of mention rank vs next-month return rank)
    rank_ic=-0.035, rank_ic_p=0.448, rank_ic_pairs=468,
    # costs
    turnover=6.8, drag_bps=4.1, net_spread_ann=-4.0, net_spread_t=-0.15,
    # random control
    random_excess_bps=-17.4,
    # synthetic controls
    syn_null_t=-0.52, syn_pos_t=14.81, syn_rev_t=-16.01,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
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

from wsb_mentions import data, strategy as st

CACHE_PATH = data.MONTHLY_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

mentions = data.mention_counts()
sig = st.mention_signal(mentions)

if HAVE_REAL:
    prices = data.fetch_meme_monthly(fetch=False, cache_path=CACHE_PATH)
    res = st.half_returns(sig, prices, q=0.50, min_stocks=6)
    spread = res["spread"].dropna()
    hype_x = (res["hype"] - res["basket"]).dropna()
    print(f"Real meme panel loaded: {prices.shape[0]} months x {prices.shape[1]} tickers")
    print(f"Portfolio months: {len(res)}  (avg {res['n_total'].mean():.0f} names, {res['n_hype'].mean():.0f} in the loud leg)")
else:
    prices = res = spread = hype_x = None
    print("No real cache -- frozen headline numbers from R dict will be used")
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
# Study 254 -- WSB-Mentions
## For the curious: does the number of r/WallStreetBets mentions call the meme move?

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*

---

In January 2021 a Reddit forum, r/WallStreetBets, lit GameStop on fire. Ever since,
a cottage industry of "mention trackers" promises an edge: **count how often a ticker
is named on WSB this month, then buy the loudest names before they run.**

We hardcode a curated monthly WSB mention-count table for a 14-name meme basket
(GME, AMC, BB, BBBY, KOSS, PLTR, ... -- including names that went to zero), join it to
real Yahoo monthly returns, and ask the only honest question: *does this month's buzz
forecast next month's return?*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## The buzz timeline: a spike, a second wave, then a long fade

The mention table encodes the canonical meme-cycle shape: the January-2021 GME/AMC
explosion, the June-2021 AMC second wave, and the slow 2022-2023 cool-off (with
BBBY's final August-2023 pump before bankruptcy).
"""),
        code("""\
fig, ax = plt.subplots(figsize=(11, 4.5))
for tk, col in [("GME", RED), ("AMC", AMBER), ("BBBY", GREY)]:
    if tk in mentions.columns:
        ax.plot(mentions.index, mentions[tk], lw=1.8, label=tk,
                color=col, marker="o", ms=3)
ax.set_title("Hardcoded WSB monthly mention counts (curated approximation)")
ax.set_ylabel("Mentions / month")
ax.legend()
plt.tight_layout()
plt.savefig("../docs/mentions.png", dpi=120, bbox_inches="tight")
plt.show()
print("Peak GME buzz:", int(mentions['GME'].max()), "mentions in", mentions['GME'].idxmax().date())
print("BBBY's last gasp:", int(mentions['BBBY'].max()), "mentions in", mentions['BBBY'].idxmax().date())
"""),

        md("""\
## The strategy in plain words

Every month-end, split the basket in half by mention count: **buy the loud half,
short the quiet half, hold one month, repeat.** If "Reddit knows", the loud half
should beat the quiet half next month.
"""),
        md("## The race: loud half vs quiet half vs the whole basket"),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

if HAVE_REAL:
    ax = axes[0]
    cum_hype   = (1 + res["hype"].dropna()).cumprod()
    cum_quiet  = (1 + res["quiet"].dropna()).cumprod()
    cum_basket = (1 + res["basket"].dropna()).cumprod()
    ax.plot(cum_hype.index,   cum_hype.values,   color=GREEN, lw=1.8, label="Loud half (high buzz)")
    ax.plot(cum_quiet.index,  cum_quiet.values,  color=RED,   lw=1.8, label="Quiet half (low buzz)")
    ax.plot(cum_basket.index, cum_basket.values, color=GREY,  lw=1.2, ls="--", label="Equal-weight basket")
    ax.set_title("Cumulative growth of $1 (price-only, delisting-pruned)")
    ax.set_ylabel("Cumulative return ($)")
    ax.legend(fontsize=9)

    ax = axes[1]
    roll = spread.rolling(6).mean() * 12 * 100
    ax.plot(roll.index, roll.values, color=GREY, lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Rolling 6-month spread (% / yr)")
    ax.set_ylabel("Annualised spread (%/yr)")
else:
    for ax in axes:
        ax.text(0.5, 0.5, f"Cache absent\\nFrozen: spread = {R['spread_ann']:+.1f}%/yr  t = {R['spread_t']:+.2f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color=GREY)
        ax.set_title("(frozen numbers)")

plt.tight_layout()
plt.savefig("../docs/equity_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Loud-minus-quiet spread: {R['spread_ann']:+.1f}%/yr = {R['spread_bps']:+.1f} bps/mo   HAC t = {R['spread_t']:+.2f}")
print(f"Hit rate: {R['spread_hit']:.1%}  (a coin flip)")
"""),

        md("""\
## The headline: the buzz tells you nothing

The loud-minus-quiet spread is **{spread:+.1f}%/yr with a HAC *t* of {t:+.2f}** -- a
statistical zero. Three things to notice:

1. **The whole basket bled.** Buying *any* of these names over 2021-2023 lost about
   **-23%/yr**. The loud half (-25.9%/yr) and the quiet half (-22.4%/yr) both
   cratered -- the mention rank just didn't sort the wreckage.

2. **The hit rate is {hit:.0%}** -- indistinguishable from a coin flip.

3. **Mentions are a lagging shadow of price.** Reddit gets loud *because* a stock
   already ran. A proxy that fires *after* the move can't forecast the next one.
""".format(spread=R['spread_ann'], t=R['spread_t'], hit=R['spread_hit'])),

        md("## Why the headline is, if anything, *too kind*"),
        code("""\
print("Two reasons the real edge is even worse than the (already-zero) headline:")
print()
print("1. SURVIVORSHIP via delisting: WISH/BBBY blew up. When a name has no price")
print("   in a month it silently drops from the short leg -- so the worst collapses")
print("   are quietly excluded, FLATTERING any 'short the hype' result.")
print()
print(f"2. COSTS: meme names are illiquid and pay heavy borrow on the short leg.")
print(f"   At 30 bps one-way the net spread is {R['net_spread_ann']:+.1f}%/yr (t={R['net_spread_t']:+.2f}).")
print()
print(f"Even the *reversal* ('short the loud names') is a zero: {R['reversal_ann']:+.1f}%/yr, t={R['reversal_t']:+.2f}.")
"""),

        md("""\
## The bottom line for the curious

The WSB mention count is **folklore, not a signal.** On a curated 36-month meme
basket it sorts next-month returns no better than chance (loud-minus-quiet
*t* = -0.14, hit rate 47%), the whole basket lost ~23%/yr regardless of buzz, and
the headline is already an *upper bound* because delisted blow-ups drop out of the
short leg.

This is a **None / Mirage**: no statistical signal, and no tradable edge once you
account for illiquidity, borrow, and the fact that mentions are a *lagging* echo of
the price move they supposedly predict.
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
# Study 254 -- WSB-Mentions
## For the quants: rank-IC, HAC tests, bootstrap CI, sub-periods, synthetic controls

*Part of [Open-Alpha-Lab](../../../README.md). See the [desk methodology](../../../METHODOLOGY.md).*
"""),
        md("## Setup"),
        code(BOOT),
        code(R_INJECT),

        md("""\
## Synthetic controls: the engine reads a planted link correctly

> 💡 *In plain words: before trusting the verdict on the real tape, we confirm the
> machinery isn't broken. We plant a known mention->return link in a synthetic
> basket and check the engine recovers it -- positive, negative, and null.*
"""),
        code("""\
# Positive control: high-buzz LEADS (premium=+0.15) -> loud-minus-quiet t >> 2
p_pos, m_pos, _ = data.synthetic_panel(premium=0.15, n_months=120, seed=254)
r_pos = st.half_returns(st.mention_signal(m_pos), p_pos, q=0.50)
s_pos = st.summarize(r_pos["spread"].dropna())
print(f"Positive control (premium=+0.15, buzz leads): spread t = {s_pos['tstat']:+.2f}")

# Reversal control: high-buzz LAGS (premium=-0.15) -> loud-minus-quiet t << -2
p_rev, m_rev, _ = data.synthetic_panel(premium=-0.15, n_months=120, seed=254)
r_rev = st.half_returns(st.mention_signal(m_rev), p_rev, q=0.50)
s_rev = st.summarize(r_rev["spread"].dropna())
print(f"Reversal control (premium=-0.15, buzz fades): spread t = {s_rev['tstat']:+.2f}")

# Null control: mentions carry no forward info (premium=0)
p_nul, m_nul, _ = data.synthetic_panel(premium=0.0, n_months=120, seed=254)
r_nul = st.half_returns(st.mention_signal(m_nul), p_nul, q=0.50)
s_nul = st.summarize(r_nul["spread"].dropna())
print(f"Null control     (premium= 0.00, no link)    : spread t = {s_nul['tstat']:+.2f}")
print("\\n-> Engine recovers sign and significance when present; reads ~zero on the null.")
"""),

        md("## Primary spec on the real tape: loud-minus-quiet, 1-month hold"),
        code("""\
if HAVE_REAL:
    s_spread = st.summarize(spread)
    s_hype   = st.summarize(res["hype"].dropna())
    s_quiet  = st.summarize(res["quiet"].dropna())
    s_basket = st.summarize(res["basket"].dropna())
    s_hype_x = st.summarize(hype_x)
    print("=== Primary spec: buy loud half, short quiet half, hold 1 month ===")
    print(f"Portfolio months: {len(res)}  |  avg names: {res['n_total'].mean():.0f}  |  loud leg: {res['n_hype'].mean():.1f}")
    print()
    print(f"LOUD  : {s_hype['mean']*12*100:+.1f}%/yr   HAC t={s_hype['tstat']:+.2f}")
    print(f"QUIET : {s_quiet['mean']*12*100:+.1f}%/yr   HAC t={s_quiet['tstat']:+.2f}")
    print(f"BASKET: {s_basket['mean']*12*100:+.1f}%/yr  (the meme basket bled regardless of buzz)")
    print(f"SPREAD: {s_spread['mean']*12*100:+.1f}%/yr = {s_spread['mean']*10000:+.1f} bps/mo  HAC t={s_spread['tstat']:+.2f}  hit={s_spread['hit_rate']:.3f}")
    print(f"LOUD excess-basket: {s_hype_x['mean']*12*100:+.1f}%/yr  HAC t={s_hype_x['tstat']:+.2f}")
    try:
        from quantlab.stats import sharpe_ci_bootstrap
        ci = sharpe_ci_bootstrap(spread, n_boot=2000, periods_per_year=12, seed=254)
        print(f"\\nBootstrap spread-Sharpe 95% CI: [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  ({ci['frac_negative']*100:.0f}% of resamples negative)")
        print("  -> CI straddles zero: no edge.")
    except ImportError:
        print(f"\\n[quantlab.stats unavailable -- frozen CI: [{R['spread_sr_lo']:+.3f}, {R['spread_sr_hi']:+.3f}]]")
else:
    print(f"Frozen: spread = {R['spread_ann']:+.1f}%/yr  HAC t = {R['spread_t']:+.2f}  hit = {R['spread_hit']:.3f}")
    print(f"Bootstrap 95% CI: [{R['spread_sr_lo']:+.3f}, {R['spread_sr_hi']:+.3f}]  ({R['spread_frac_neg']*100:.0f}% negative)")
"""),

        md("""\
## The cleanest test: pooled rank-IC

> 💡 *In plain words: forget portfolios for a second. Across every (stock, month)
> pair, does a higher mention rank line up with a higher next-month return rank?
> That is the Spearman rank information coefficient (rank-IC). If buzz forecasts
> returns, it should be reliably positive.*
"""),
        code("""\
import scipy.stats as ss
fwd = prices.pct_change().shift(-1) if HAVE_REAL else None
if HAVE_REAL:
    X, Y = [], []
    for t in sig.index:
        if t not in fwd.index: continue
        s = sig.loc[t].dropna(); f = fwd.loc[t].dropna()
        c = s.index.intersection(f.index)
        if len(c) < 6: continue
        X.append(s.loc[c].rank()); Y.append(f.loc[c].rank())
    X = pd.concat(X); Y = pd.concat(Y)
    rho, pval = ss.spearmanr(X, Y)
    print(f"Pooled rank-IC (Spearman): {rho:+.3f}  p = {pval:.3f}  over {len(X)} (stock, month) pairs")
else:
    rho, pval = R['rank_ic'], R['rank_ic_p']
    print(f"Frozen pooled rank-IC: {rho:+.3f}  p = {pval:.3f}  over {R['rank_ic_pairs']} pairs")
print("-> Indistinguishable from zero. Mention rank does not order next-month returns.")
"""),

        md("## Sub-periods: no edge in any single year either"),
        code("""\
if HAVE_REAL:
    sub_stats = []
    for label in ["2021", "2022", "2023"]:
        sub = spread[label:label]
        s = st.summarize(sub)
        sub_stats.append({"period": label, "mean_ann": s["mean"]*12*100, "tstat": s["tstat"], "n": s["n"]})
        print(f"  {label}: mean={s['mean']*12*100:+.1f}%/yr  HAC t={s['tstat']:+.2f}  n={s['n']}")
    sts = pd.DataFrame(sub_stats)
else:
    sts = pd.DataFrame([
        {"period": "2021", "mean_ann": R["sub_2021_mean"], "tstat": R["sub_2021_t"], "n": R["sub_2021_n"]},
        {"period": "2022", "mean_ann": R["sub_2022_mean"], "tstat": R["sub_2022_t"], "n": R["sub_2022_n"]},
        {"period": "2023", "mean_ann": R["sub_2023_mean"], "tstat": R["sub_2023_t"], "n": R["sub_2023_n"]},
    ])
    for _, r in sts.iterrows():
        print(f"  {r['period']}: mean={r['mean_ann']:+.1f}%/yr  HAC t={r['tstat']:+.2f}  n={int(r['n'])}")

fig, ax = plt.subplots(figsize=(8, 4))
colors_sub = [GREEN if abs(t) >= 2 else GREY for t in sts["tstat"]]
ax.bar(sts["period"], sts["tstat"], color=colors_sub)
ax.axhline(2.0, color="black", ls="--", lw=0.9, label="|t|=2 inference bar")
ax.axhline(-2.0, color="black", ls="--", lw=0.9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Sub-period HAC t-stat (none clears |t|=2)")
ax.set_ylabel("HAC t-stat")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""),

        md("## Turnover, costs, and the random-portfolio null"),
        code("""\
if HAVE_REAL:
    drag = st.turnover_cost_drag(sig, prices, q=0.50, one_way_bps=30.0)
    avg_turnover = drag.mean() / 0.006
    net_spread = spread - drag.reindex(spread.index).fillna(drag.mean())
    s_net = st.summarize(net_spread)
    print(f"Avg monthly loud-leg turnover: {avg_turnover:.1%}")
    print(f"Avg drag @30bps one-way (illiquid + borrow): {drag.mean()*10000:.1f} bps/mo")
    print(f"Net spread @30bps: {s_net['mean']*12*100:+.1f}%/yr  HAC t={s_net['tstat']:+.2f}")
    print()
    rand = st.random_portfolio_returns(sig, prices, q=0.50, n_draws=200, seed=254)
    print(f"Random-half excess vs basket: mean={rand.mean()*10000:+.1f} bps  (null centred near zero)")
    print(f"  -> The mention-sorted loud leg is no better than a random half of the basket.")
else:
    print(f"Frozen: turnover={R['turnover']:.1f}%  drag={R['drag_bps']:.1f}bps/mo")
    print(f"Net spread @30bps: {R['net_spread_ann']:+.1f}%/yr  HAC t={R['net_spread_t']:+.2f}")
    print(f"Random-half excess: {R['random_excess_bps']:+.1f} bps")
"""),

        md("## Verdict"),
        code("""\
print("=== Study 254 -- WSB-Mentions ===")
print()
print(f"Signal: NONE")
print(f"  Loud-minus-quiet spread HAC t = {R['spread_t']:+.2f} (a statistical zero)")
print(f"  Pooled rank-IC = {R['rank_ic']:+.3f} (p={R['rank_ic_p']:.2f}); hit rate {R['spread_hit']:.0%}")
print(f"  Bootstrap CI straddles zero; no sub-period clears |t|=2")
print(f"  Synthetic controls confirm the engine WOULD see a real link (t={R['syn_pos_t']:+.1f}/{R['syn_rev_t']:+.1f})")
print()
print(f"Tradability: MIRAGE")
print(f"  Net spread @30bps one-way = {R['net_spread_ann']:+.1f}%/yr (t={R['net_spread_t']:+.2f})")
print(f"  Illiquid meme names, heavy short borrow; loud leg ~ a random half")
print()
print("Endogeneity (named on Signal): mentions are CONTEMPORANEOUS -- Reddit gets")
print("  loud BECAUSE the price already moved, so the proxy can't lead the move.")
print("Survivorship (named on Signal): delisted blow-ups (WISH/BBBY) drop from the")
print("  short leg, so the headline is already an UPPER BOUND on any reversal edge.")
print()
print("Bottom line: None/Mirage -- WSB mention counts are folklore, not a forecast.")
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
    import subprocess, sys
    path = os.path.join(HERE, filename)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            "--ExecutePreprocessor.startup_timeout=180",
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
