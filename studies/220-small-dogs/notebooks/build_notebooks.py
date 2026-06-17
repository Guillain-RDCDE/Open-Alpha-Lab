"""Build the two narrative notebooks for Study 220 (Small Dogs of the Dow).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks walk the seven desk beats at two altitudes. Figures are computed live from the
``small_dogs`` package on the cached real panel; the frozen ``R`` dict mirrors
``docs/results.md`` so the two never drift.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    as_of="2026-06-13", fp="6dfd9e54cba6", n_years=26, first=2000, last=2025,
    n_tickers=46, n_days=6673,
    # Small Dogs
    small_cagr=10.57, small_total=13.64,
    small_gap_dow=3.04, small_mean_exc=2.88, small_win=65, small_win_n=17,
    small_hac_t=1.82, small_boot_lo=-0.27, small_boot_hi=5.63, small_boot_p=0.07,
    small_alpha=5.26, small_alpha_t=2.68, small_beta=0.72, small_r2=0.59,
    # Small vs Dogs incremental
    small_vs_dogs_mean=2.01, small_vs_dogs_t=2.40, small_vs_dogs_lo=0.41, small_vs_dogs_hi=3.71,
    # Full Dogs
    dogs_cagr=8.51, dogs_total=8.36, dogs_gap_dow=0.98,
    dogs_hac_t=0.59,
    # Foolish Four
    f4_cagr=7.67, f4_total=6.84, f4_gap_dow=0.14,
    f4_hac_t=-0.09, f4_mean_exc=-0.11,
    # Dow
    dow_cagr=7.54, dow_total=7.61,
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study root (small_dogs/)
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from small_dogs import data, strategy

AS_OF = "2026-06-13"
CACHE = os.path.abspath(os.path.join("..", "_cache"))
panel = data.load_panel(end=AS_OF, cache_dir=CACHE)
ann = strategy.annual_returns_from_panel(panel, 2000, 2025, cost_bps=20.0)
summ_small = strategy.summarize(ann, col="small_dogs", bench="dow")
summ_dogs  = strategy.summarize(ann, col="dogs", bench="dow")
summ_f4    = strategy.summarize(ann, col="foolish4", bench="dow")
exc_small = (ann["small_dogs"] - ann["dow"]).to_numpy()
exc_dogs  = (ann["dogs"] - ann["dow"]).to_numpy()
exc_f4    = (ann["foolish4"] - ann["dow"]).to_numpy()
hac_small = strategy.hac_tstat(exc_small)
hac_dogs  = strategy.hac_tstat(exc_dogs)
hac_f4    = strategy.hac_tstat(exc_f4)
boot_small = strategy.block_bootstrap_ci(exc_small)
ab_small = strategy.alpha_beta(ann, strat_col="small_dogs", bench_col="dow")
print(f"Dow panel: {panel['prices'].shape[1]} tickers x {len(panel['prices']):,} days  fingerprint={data.fingerprint(panel)}")
print(f"Small Dogs CAGR {summ_small['strat_cagr']*100:.2f}%  vs Dow {summ_small['bench_cagr']*100:.2f}%  gap {summ_small['cagr_gap']*100:+.2f} pts/yr  ({summ_small['n_years']} yrs)")
print(f"Dogs      CAGR {summ_dogs['strat_cagr']*100:.2f}%   HAC t {hac_dogs:+.2f}")
print(f"Small Dogs HAC t {hac_small:+.2f}   alpha {ab_small['alpha']*100:+.2f}%/yr (t {ab_small['alpha_t']:+.2f})   beta {ab_small['beta']:.2f}")
print(f"Foolish 4  HAC t {hac_f4:+.2f}")
"""


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


HERO_BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
)


def build_curious():
    cells = [
        md(
            "# Small Dogs of the Dow\n"
            "### Do the 5 cheapest Dogs — or the Foolish Four — beat the full ten?\n\n"
            + HERO_BADGES +
            "The **Dogs of the Dow** is the simple rule of buying the ten highest-yielding Dow "
            "stocks each January. A popular 1990s upgrade claimed you could do even better: take "
            "only the **five cheapest by share price** (the 'Small Dogs' or 'Low-5'), and better "
            "still with the **'Foolish Four'** — skip the #1 cheapest Dog, buy the next four. "
            "We test all three variants on the same honest 26-year tape and discover that the "
            "Small Dogs look interesting on the surface, the Foolish Four is pure noise, and the "
            "'edge' from a price filter is mostly a crude small-cap tilt you can buy more cheaply.\n\n"
            "> 📓 **This is the plain-language layer.** The *t*-stats, multiple-comparison "
            "corrections and risk adjustments live in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n>\n"
            "> ⚠️ **Not investment advice.** Educational, reproducible research. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(
            "## The answer first\n\n"
            f"| Question | Answer |\n|---|---|\n"
            f"| Did the Small Dogs beat the Dow over {R['n_years']} honest years? | **Yes, by +{R['small_gap_dow']:.1f} pts/yr** "
            f"(CAGR **{R['small_cagr']}%** vs **{R['dow_cagr']}%**) |\n"
            f"| Is that lead statistically real? | **Borderline no** — HAC *t* = **{R['small_hac_t']}** "
            f"(bar is 2.0); CI **[{R['small_boot_lo']}%, +{R['small_boot_hi']}%]**, *p* = {R['small_boot_p']} |\n"
            f"| Did the Foolish Four work? | **No** — essentially flat vs the Dow "
            f"(+{R['f4_gap_dow']:.2f} pts/yr, *t* = {R['f4_hac_t']}) |\n"
            f"| What explains the Small Dogs' edge? | Mostly a **low-price / low-beta tilt** "
            f"(beta {R['small_beta']}) — the same thing a small-cap-value ETF gives you cheaply |"
        ),
        md(
            "## 1 · The claim\n\n"
            "Michael O'Higgins showed in *Beating the Dow* (1991) that picking the ten highest-yield "
            "Dow stocks each January beat the index. His 'Low-5' variant: of those ten, take only "
            "the **five with the lowest absolute share price**. The Motley Fool's 'Foolish Four' "
            "(1996) went further: skip #1 cheapest, buy #2-#5. The intuition — low share prices "
            "signal beaten-down value — sounds coherent. But there's a catch: the Foolish Four was "
            "discovered by looking at a historical table and noting which variant looked best. That "
            "is not a discovery; it is a data artefact waiting to collapse."
        ),
        md(
            "## 2 · So what?\n\n"
            "If a price-filter on top of the Dogs adds genuine value, it's telling you something "
            "real about which beaten-down Dow names recover. If it's curve-fitting on a 20-year "
            "window, it vanishes the moment you start trading it live — which is exactly what the "
            "Motley Fool found in 2000 when they retired the Foolish Four and called it a mistake."
        ),
        md(
            "## 3 · How would we even know?\n\n"
            "We run all three variants (Dogs-10, Small Dogs-5, Foolish Four) on the **exact same "
            "26-year point-in-time tape** — the same membership timeline from Study 88, the same "
            "total-return prices, the same 20 bps/turnover cost, the same DIA benchmark. The only "
            "thing that changes each year is *which* of the ten Dogs we hold. Any difference in "
            "outcomes is attributable to the sub-selection rule alone."
        ),
        md("## 4 · The teardown — let's actually look\n\nThree equity curves, $1 invested in 2000:"),
        code(
            "eq_small = (1 + ann['small_dogs']).cumprod()\n"
            "eq_dogs  = (1 + ann['dogs']).cumprod()\n"
            "eq_f4    = (1 + ann['foolish4']).cumprod()\n"
            "eq_dow   = (1 + ann['dow']).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            "ax.plot(eq_small.index, eq_small.values, marker='o', lw=1.8, label=f\"Small Dogs-5 ({summ_small['strat_cagr']*100:.1f}%/yr)\")\n"
            "ax.plot(eq_dogs.index, eq_dogs.values, marker='s', lw=1.6, label=f\"Dogs-10 ({summ_dogs['strat_cagr']*100:.1f}%/yr)\")\n"
            "ax.plot(eq_f4.index, eq_f4.values, lw=1.3, ls='--', label=f\"Foolish Four ({summ_f4['strat_cagr']*100:.1f}%/yr)\")\n"
            "ax.plot(eq_dow.index, eq_dow.values, lw=1.2, ls=':', alpha=.9, label=f\"Dow / DIA ({summ_small['bench_cagr']*100:.1f}%/yr)\")\n"
            "ax.set_yscale('log'); ax.set_ylabel('Growth of $1 (log)'); ax.set_xlabel('rebalance year')\n"
            "ax.set_title('Small Dogs / Foolish Four / full Dogs / Dow — 26 years, total return, net 20 bps')\n"
            "ax.legend(); plt.show()"
        ),
        md(
            f"The Small Dogs line ends highest — $**{R['small_total']:.2f}** vs $**{R['dogs_total']:.2f}** "
            f"for the Dogs and $**{R['dow_total']:.2f}** for the Dow. But the Foolish Four ($**{R['f4_total']:.2f}**) "
            "lands *below* the Dow, the strategy that was supposed to be even better. Let's check how much "
            "of the Small Dogs' lead is annual-consistent vs lucky streaks."
        ),
        code(
            "fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)\n"
            "for ax, col, label in zip(axes,\n"
            "        ['small_dogs', 'dogs', 'foolish4'],\n"
            "        ['Small Dogs vs Dow', 'Dogs vs Dow', 'Foolish Four vs Dow']):\n"
            "    exc = (ann[col] - ann['dow'])*100\n"
            "    colors = ['#2ea44f' if e > 0 else '#c0392b' for e in exc]\n"
            "    ax.bar(ann.index, exc, color=colors)\n"
            "    ax.axhline(0, color='k', lw=.8)\n"
            "    ax.axhline(exc.mean(), color='C0', ls='--', label=f'mean {exc.mean():+.2f}%/yr')\n"
            "    ax.set_ylabel('Annual excess (%)')\n"
            "    ax.set_title(label); ax.legend()\n"
            "axes[-1].set_xlabel('year'); plt.tight_layout(); plt.show()"
        ),
        md(
            "Notice the Foolish Four's bars: they bounce around zero with no trend, confirming "
            "the strategy is noise. The Small Dogs' bars lean positive but contain several large "
            "negative years — the mean is real-looking, but so is the variance around it."
        ),
        md("Now what does 'low-priced Dogs' actually buy you in a crisis? Let's check the Jan-2009 basket."),
        code(
            "from small_dogs.strategy import trailing_yield\n"
            "rebal = pd.Timestamp('2008-12-31')\n"
            "members = data.members_asof(rebal)\n"
            "yy, pxr = {}, {}\n"
            "for t in members:\n"
            "    if t in panel['prices'].columns:\n"
            "        dv = panel['divs'][t].dropna() if t in panel['divs'].columns else pd.Series(dtype=float)\n"
            "        v = trailing_yield(panel['prices'][t].dropna(), dv, rebal)\n"
            "        seg = panel['prices'][t].dropna().loc[:rebal]\n"
            "        if np.isfinite(v) and v > 0 and not seg.empty:\n"
            "            yy[t] = v; pxr[t] = float(seg.iloc[-1])\n"
            "dogs_top10 = sorted(yy, key=lambda k: yy[k], reverse=True)[:10]\n"
            "by_price = sorted(dogs_top10, key=lambda k: pxr[k])\n"
            "print('Jan-2009 Dogs basket: yield / share price (cheapest first)')\n"
            "for t in by_price:\n"
            "    mark = ' << Small Dog' if t in by_price[:5] else ''\n"
            "    print(f'  {t:5s}  yield {yy[t]*100:5.1f}%  price ${pxr[t]:7.2f}{mark}')"
        ),
        md(
            "The January-2009 Small Dogs basket is a concentration of financials in free-fall: "
            "Citigroup and Bank of America top the yield list (both about to cut dividends to near "
            "zero) and are also among the cheapest by share price — so both filters agree on "
            "picking the most distressed names. The 'value' signal here is a dividend trap. The "
            "Small Dogs didn't just pick beaten-down names; they over-concentrated in them."
        ),
        md(
            "## 5 · The verdict\n\n"
            f"- **Small Dogs beat the Dow by +{R['small_gap_dow']:.1f} pts/yr over 26 years** — "
            f"but the HAC *t* sits at **{R['small_hac_t']}**, just below the bar of 2.0 (*p* = "
            f"{R['small_boot_p']}). The CI straddles zero at [{R['small_boot_lo']}%, "
            f"+{R['small_boot_hi']}%]. → **Signal: Weak.**\n"
            f"- **The Foolish Four earned nothing** (+{R['f4_gap_dow']:.2f} pts/yr, "
            f"*t* = {R['f4_hac_t']}) — the skip-the-cheapest quirk was pure curve-fitting. → Bust.\n"
            "- **The price-filter tilt is a crude small-cap proxy.** A low absolute share price "
            "within the Dow 30 is not an independent signal — it is a rough proxy for the "
            "size/value factor, replicable far more cheaply with a small-cap-value ETF.\n"
            "- **Tradability: Mirage.** A 5-name equal-weight basket of the cheapest blue chips "
            "is highly concentrated, carries dividend-trap risk (2009 again), and the data-mining "
            "history means any statistical flicker is suspect. → **Tradability: Mirage.**"
        ),
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The trade is technically trivial — five blue-chip Dow names, annual rebalance. But "
            "the economics don't work: (1) the edge doesn't clear a significance test; (2) it's "
            "a 5-name portfolio with massive single-stock risk; (3) the Foolish Four — the variant "
            "that attracted real money in the late 1990s — collapsed immediately in live trading; "
            "(4) you're buying a small-cap tilt wrapped in a dividend-trap screen. You can own a "
            "small-cap-value ETF for 10-15 bps/year without any of these drawbacks."
        ),
        md(
            "## 7 · Going further\n\n"
            "- Replace absolute price with **market-cap rank** within the Dogs — does the "
            "size tilt read cleaner as an explicit factor?\n"
            "- Apply a **dividend-trap filter** (forward earnings yield or payout-ratio cap) "
            "before the price screen — does avoiding the 2009 financials recover a real edge?\n"
            "- Test the same price-filter idea on a **wider universe** (S&P 100 high-dividend "
            "names) to see whether the residual tilt is Dogs-specific or factor-general.\n"
            "- See [Study 88 — Dogs-of-the-Dow](../../../88-dogs-of-the-dow/) for the parent "
            "strategy and the full point-in-time membership teardown."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Small Dogs of the Dow — quantitative teardown\n"
            "### Three-strategy simultaneous test · HAC + block-bootstrap · alpha-vs-beta · data-mining adjustment\n\n"
            + HERO_BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We run the Dogs-10, Small Dogs-5, and Foolish Four simultaneously on the same "
            "26-year point-in-time tape and apply the inference machinery honestly: HAC t-stats, "
            "circular block bootstrap, alpha-vs-beta OLS, and a multiple-comparison framing that "
            "accounts for the fact that the price filter was discovered by fitting to this same "
            "historical window.\n\n"
            "> ⚠️ **Not investment advice.** Dow components, total-return adjusted + dividend stream "
            "(`yfinance`), point-in-time membership from Study 88; benchmark DIA total return; "
            "20 bps/turnover. Sources in [`docs/references.md`](../docs/references.md), "
            "reproducible run in [`docs/results.md`](../docs/results.md).\n>\n"
            "> 💡 **`💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Why |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Small Dogs +{R['small_gap_dow']:.1f} pts/yr over {R['n_years']} yrs, "
            f"but **HAC *t* = {R['small_hac_t']}** (bar is 2.0); CI [{R['small_boot_lo']}%, "
            f"+{R['small_boot_hi']}%] barely clips zero (*p* = {R['small_boot_p']}). "
            f"Incremental test (Small vs Dogs) *t* = +{R['small_vs_dogs_t']:.2f} but "
            f"carries data-mining debt. Foolish Four HAC *t* = {R['f4_hac_t']:.2f} — flat noise. |\n"
            f"| **Tradability** | `MIRAGE` | 5-name equal-weight basket of cheapest blue chips; "
            f"low-beta tilt (beta {R['small_beta']}) replicable with a small-cap-value ETF; "
            f"dividend-trap risk (2009 financials); multiple-comparison penalty; in-sample origins. |"
        ),
        md(
            "## 1 · Hypotheses\n\n"
            "- **H₁ (price-filter tilt):** the low absolute share-price filter within Dogs picks "
            "names with a higher size/value tilt — a known risk premium.\n"
            "- **H₂ (alpha claim):** the Small Dogs beat the Dow on total return, net, out-of-sample.\n"
            "- **H₃ (data-mining null):** the Foolish Four (skip #1, keep #2-#5) and the Small Dogs "
            "were found by sorting historical tables — without a prior theory, any apparent edge is "
            "a candidate for the data-mining null.\n\n"
            "H₁ is almost certainly true (the low-price filter is a crude size proxy). H₂ is "
            "borderline. H₃ cautions against celebrating H₂."
        ),
        md("## 2 · So what? — what rides on each answer\n\nIf H₂ held for Small Dogs with honest "
           "inference, a one-sentence price rule would add ~3 pts/yr to an already sub-par "
           "Dogs strategy. But the Foolish Four's collapse in live trading is the out-of-sample "
           "data point we already have — a near-identical in-sample discovery that went to zero "
           "the moment it got deployed."),
        md("## 3 · Protocol\n\nPoint-in-time membership timeline (Study 88) · annual total-return "
           "engine vs DIA · 20 bps/turnover · HAC t and circular block bootstrap on three "
           "simultaneous strategy series · alpha-vs-beta OLS · multiple-comparison framing."),
        md("## 4 · The annual table\n\nDogs-10, Small Dogs-5, Foolish Four, and Dow:"),
        code(
            "show = ann[['dogs','small_dogs','foolish4','dow','excess_dogs','excess_small','excess_f4']].copy()\n"
            "for c in show.columns: show[c] = (show[c]*100).round(2)\n"
            "show"
        ),
        md("**Inference — each strategy vs Dow (HAC t and circular block bootstrap):**"),
        code(
            "boot_dogs = strategy.block_bootstrap_ci(exc_dogs)\n"
            "boot_f4   = strategy.block_bootstrap_ci(exc_f4)\n"
            "lo_s, hi_s, p_s = boot_small\n"
            "lo_d, hi_d, p_d = boot_dogs\n"
            "lo_f, hi_f, p_f = boot_f4\n"
            "print('Strategy     | mean exc | HAC t | 95% CI            | p')\n"
            "print('-------------|----------|-------|-------------------|------')\n"
            "print(f'Small Dogs-5 | {exc_small.mean()*100:+.2f}%/yr | {hac_small:+.2f}  | [{lo_s*100:+.2f}%, {hi_s*100:+.2f}%] | {p_s:.2f}')\n"
            "print(f'Dogs-10      | {exc_dogs.mean()*100:+.2f}%/yr | {hac_dogs:+.2f}  | [{lo_d*100:+.2f}%, {hi_d*100:+.2f}%] | {p_d:.2f}')\n"
            "print(f'Foolish Four | {exc_f4.mean()*100:+.2f}%/yr | {hac_f4:+.2f}  | [{lo_f*100:+.2f}%, {hi_f*100:+.2f}%] | {p_f:.2f}')\n"
            "print()\n"
            "print('=> Small Dogs t = 1.82 is below the |t|>=2 bar; Foolish Four is pure noise.')"
        ),
        md(
            "> 💡 **In plain words:** none of the three strategies clears the |t| ≥ 2 bar on its "
            "annual excess vs the Dow. The Small Dogs comes closest (t = 1.82) but misses. The "
            "Foolish Four is indistinguishable from zero."
        ),
        md("**Incremental test: Small Dogs vs full Dogs — does the price filter add anything?**"),
        code(
            "exc_sv_d = (ann['small_dogs'] - ann['dogs']).to_numpy()\n"
            "t_svd = strategy.hac_tstat(exc_sv_d)\n"
            "lo_svd, hi_svd, p_svd = strategy.block_bootstrap_ci(exc_sv_d)\n"
            "exc_fvd = (ann['foolish4'] - ann['dogs']).to_numpy()\n"
            "t_fvd = strategy.hac_tstat(exc_fvd)\n"
            "lo_fvd, hi_fvd, p_fvd = strategy.block_bootstrap_ci(exc_fvd)\n"
            "print(f'Small Dogs minus Dogs: mean {exc_sv_d.mean()*100:+.2f}%/yr  t {t_svd:+.2f}  CI [{lo_svd*100:+.2f}%, {hi_svd*100:+.2f}%]  p={p_svd:.2f}')\n"
            "print(f'Foolish Four minus Dogs: mean {exc_fvd.mean()*100:+.2f}%/yr  t {t_fvd:+.2f}  CI [{lo_fvd*100:+.2f}%, {hi_fvd*100:+.2f}%]  p={p_fvd:.2f}')\n"
            "print()\n"
            "print('=> Small Dogs clears t>2 on the incremental test, but with data-mining baggage.')\n"
            "print('=> Foolish Four subtracts from the Dogs — the skip-the-cheapest quirk is negative.')"
        ),
        md(
            "> 💡 **In plain words:** the price filter does appear to add ~2 pts/yr *on top of* the "
            "Dogs in this sample. But because the filter was designed by looking at this same data, "
            "a Bonferroni-style penalty would reduce the effective significance. The Foolish Four's "
            "negative contribution is the cautionary out-of-sample data point we already have."
        ),
        md("**Alpha-vs-beta: is the Small Dogs' edge real alpha or a low-beta tilt?**"),
        code(
            "x = ann['dow'].to_numpy()*100; y = ann['small_dogs'].to_numpy()*100\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "fig, axx = plt.subplots(figsize=(8, 6))\n"
            "axx.scatter(x, y, s=40, alpha=.8)\n"
            "axx.scatter(ann['dow'].to_numpy()*100, ann['dogs'].to_numpy()*100, s=20, alpha=.4, marker='^', label='Dogs-10')\n"
            "axx.plot(xs, ab_small['alpha']*100 + ab_small['beta']*xs, 'C3', lw=2,\n"
            "    label=f\"Small Dogs: alpha {ab_small['alpha']*100:+.2f}%/yr (t {ab_small['alpha_t']:+.2f}), beta {ab_small['beta']:.2f}\")\n"
            "axx.plot(xs, xs, 'k--', lw=1, alpha=.6, label='y = x (Dogs = Dow)')\n"
            "axx.set_xlabel('Dow annual return (%)'); axx.set_ylabel('Strategy annual return (%)')\n"
            "axx.set_title(f\"Alpha-vs-beta: R2 = {ab_small['r2']:.2f}\"); axx.legend(); plt.show()\n"
            "print(f\"Small Dogs alpha {ab_small['alpha']*100:+.2f}%/yr (t {ab_small['alpha_t']:+.2f})   beta {ab_small['beta']:.2f}   R2 {ab_small['r2']:.2f}\")\n"
            "print('Note: alpha t clears 2.0, but it is inflated by the low beta (0.72 vs 1.0).')"
        ),
        md(
            "> 💡 **In plain words:** the Small Dogs' alpha *t* of 2.68 clears the bar, but it's "
            "driven substantially by the fact that the basket has a beta of 0.72 — it's a lower-risk "
            "slice of the Dow that *looks* like it generates alpha in an OLS regression. The 5 "
            "cheapest Dow names skew toward the least-fashionable, highest-dividend, lowest-market-cap "
            "corner of the index — a well-known value/size tilt you were always going to be paid for."
        ),
        md(
            "## 5 · The verdict\n\n"
            f"Signal `WEAK`: Small Dogs gap +{R['small_gap_dow']:.1f} pts/yr, "
            f"HAC *t* {R['small_hac_t']} (bar is 2.0), CI [{R['small_boot_lo']}%, "
            f"+{R['small_boot_hi']}%]. Incremental t vs Dogs = +{R['small_vs_dogs_t']:.2f} "
            f"carries data-mining debt. Foolish Four HAC *t* = {R['f4_hac_t']:.2f} — pure noise. "
            f"Alpha *t* = {R['small_alpha_t']:.2f} clears bar but is inflated by low beta "
            f"({R['small_beta']}). Tradability `MIRAGE`: 5-name concentration, dividend-trap "
            "exposure, data-mining history, and a size tilt replicable far more cheaply elsewhere."
        ),
        md(
            "## 6 · Could you trade it?\n\n"
            "The mechanical rule is trivial — five Dow stocks per year. The economic case isn't:\n\n"
            "1. **The HAC t on the raw excess is 1.82** — below the house bar, especially with "
            "three tested variants (Bonferroni would push the effective p further from significance).\n"
            "2. **The Foolish Four already collapsed in live trading** (Motley Fool retired it "
            "in 2000). That is the out-of-sample test we already have for in-sample price optimisation.\n"
            "3. **5-name equal-weight Dow basket:** idiosyncratic risk is enormous. One Citi or "
            "BofA-style event in the basket wipes the annual excess multiple times over.\n"
            "4. **The low-beta tilt is replicable** with a dividend-value or small-cap-value "
            "ETF at 10-15 bps/year with 500-name diversification."
        ),
        md(
            "## 7 · Going further\n\n"
            "- Test an **explicit size-factor control**: regress Small Dogs excess on Dow "
            "excess and a small-cap factor; does the alpha survive?\n"
            "- A **dividend-payout-ratio cap** (exclude names whose yield > 2 × the median "
            "payout ratio of their earnings) to neutralise the dividend trap — does this "
            "recover a real signal?\n"
            "- The **Bonferroni / Holm correction** on all three tested variants simultaneously: "
            "compute the adjusted p-values for Dogs, Small Dogs, and Foolish Four and report "
            "whether any survive at the 5% family-wise error rate.\n"
            "- See [Study 88 — Dogs-of-the-Dow](../../../88-dogs-of-the-dow/) for the parent "
            "study and [Study 196 — Long-Term-Reversal](../../../196-long-term-reversal/) for "
            "the DeBondt-Thaler mean-reversion lens on beaten-down blue chips."
        ),
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
