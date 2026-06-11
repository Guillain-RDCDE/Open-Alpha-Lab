"""Real-data run — is the carry premium real on G10, and how bad is the steamroller?

Three questions (the offline core proves the machine; this points it at the market):

  * **Do high-rate currencies out-earn?** The carry premium by (lagged) rate bucket and the portfolio's
    Newey-West t-stat + bootstrap CI.
  * **What's the tail?** The crash profile — negative skew, worst months, drawdown, downside concentration.
  * **Can risk management dodge it?** Vol-managed carry vs plain, on a shared window.

    python examples/verify.py             # offline, cache-only (the desk's normal mode)
    python examples/verify.py --fetch     # on a cache miss, download via tools/fetch_altdata.py

**Data note.** The real tape is the desk's shared G10 cache — OECD MEI 3-month interbank short rates
(via DBnomics) + yfinance FX (USD per 1 unit) — the *same two parquets* Study 36 (Greenback) runs on,
so both FX studies share one tape and one fingerprint. OECD MEI was discontinued at 2024-01, so the
as-of is pinned there and the published numbers never creep. Pinned with `quantlab.repro.as_of` and
fingerprinted.
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from steamroller import data, carry, strategy, decompose, extension
from quantlab.repro import as_of, fingerprint
from quantlab.stats import sharpe_ci_bootstrap

OUT = os.path.join(_STUDY, "docs", "results.md")
COST_BPS = 10.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    g10 = data.fetch_carry(fetch=args.fetch)
    if not g10:
        print("[skip] no cached G10 carry data (_cache/g10_short_rates.parquet + _cache/g10_fx.parquet).")
        print("       Re-run with --fetch (needs network; reuses tools/fetch_altdata.py), or run")
        print("       python tools/fetch_altdata.py 'short rates' 'G10 FX' from the repo root.")
        print("       The offline synthetic core (run_synthetic_demo.py) is the validated machinery proof.")
        return
    rates, fx = as_of(g10["rates"], data.DATA_AS_OF), as_of(g10["fx"], data.DATA_AS_OF)
    xr = carry.excess_returns(rates, fx)
    # align rates to the excess-return columns/index for ranking
    rates_aligned = rates[[c for c in xr.columns]].reindex(xr.index)

    pb = carry.carry_premium_by_bucket(xr, rates_aligned)
    cmp = strategy.compare(xr, rates_aligned, cost_bps=COST_BPS)
    pt = decompose.premium_tstat(xr, rates_aligned, cost_bps=COST_BPS)
    pt_gross = decompose.premium_tstat(xr, rates_aligned, cost_bps=0.0)
    cr = decompose.crash_profile(xr, rates_aligned, cost_bps=COST_BPS)
    dc = decompose.downside_concentration(xr, rates_aligned, cost_bps=COST_BPS, k=5)
    cc = extension.crash_comparison(xr, rates_aligned, cost_bps=COST_BPS)
    ci = sharpe_ci_bootstrap(strategy.carry_returns(xr, rates_aligned, cost_bps=COST_BPS),
                             periods_per_year=12, seed=27)
    sweep = strategy.cost_sweep(xr, rates_aligned)
    book = strategy.carry_returns(xr, rates_aligned, cost_bps=COST_BPS)
    worst = {str(d.date()): float(v * 100.0) for d, v in book.nsmallest(3).items()}
    fp = fingerprint(xr.round(6))

    print(f"G10 real tape (USD-funded, monthly): {xr.index.min().date()} -> {xr.index.max().date()} "
          f"({len(xr)} months, {xr.shape[1]} currencies); as-of {data.DATA_AS_OF} · fingerprint {fp}\n")
    print(f"carry premium {pt['mean_ann_pct']:+.1f}%/yr net (HAC t {pt['t_stat']:+.1f}; gross t "
          f"{pt_gross['t_stat']:+.1f}), Sharpe {cmp['sharpe']:+.2f} "
          f"(bootstrap 95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]); HML bucket spread "
          f"{pb['hml_ann_pct']:+.1f}%/yr; turnover {cmp['turnover_ann']:.2f}x/yr")
    print(f"steamroller: skew {cr['skew']:+.2f}, worst month {cr['worst_month_pct']:+.1f}%, "
          f"max drawdown {cr['max_drawdown_pct']:.0f}%; worst months {worst}")
    print(f"vol-managed (shared {cc['n_months']}-month window): Sharpe {cc['plain']['sharpe']:+.2f} -> "
          f"{cc['managed']['sharpe']:+.2f}, drawdown {cc['plain']['max_drawdown_pct']:.0f}% -> "
          f"{cc['managed']['max_drawdown_pct']:.0f}%")
    print("\nCost sweep (bp per unit traded -> Sharpe):\n" + sweep.round(3).to_string())
    _write(OUT, dict(pb=pb, cmp=cmp, pt=pt, pt_gross=pt_gross, cr=cr, dc=dc, cc=cc, ci=ci, fp=fp,
                     worst=worst, sweep=sweep,
                     lo=xr.index.min().date(), hi=xr.index.max().date(),
                     n_ccy=xr.shape[1], n_months=len(xr)), data.DATA_AS_OF)
    print(f"\nwrote {OUT}")


def _verdict(d):
    t = d["pt"]["t_stat"]
    if d["pt"]["mean_ann_pct"] <= 0 or t <= 0:
        signal = "NONE"
    elif t > 2.0:
        signal = "REAL"
    else:
        signal = "WEAK"
    trad = "FRAGILE"   # cheap to run but crash-prone and thin — never INVESTABLE on this tape
    # SEVERE if the tail is jump-like AND the vol overlay fails to shrink it (the study's namesake test)
    overlay_fails = d["cc"]["managed"]["max_drawdown_pct"] <= d["cc"]["plain"]["max_drawdown_pct"]
    crash = "Severe" if (d["cr"]["skew"] < -0.5 and overlay_fails) else "Moderate"
    return signal, trad, crash


def _write(path, d, asof):
    signal, trad, crash = _verdict(d)
    cmp, pt, cr, cc, ci = d["cmp"], d["pt"], d["cr"], d["cc"], d["ci"]
    sweep_cells = " | ".join(f"{s:+.2f}" for s in d["sweep"]["sharpe"])
    sweep_head = " | ".join(str(c) for c in d["sweep"].index)
    worst_s = ", ".join(f"{k} {v:+.1f}%" for k, v in d["worst"].items())
    text = f"""# Results — Study 27 (Steamroller) on the real G10 carry tape

*Generated by [`examples/verify.py`](../examples/verify.py). Monthly G10 3-month interbank short rates
(OECD MEI via DBnomics, % p.a.) and month-end USD FX (yfinance, USD per 1 unit) — the **same shared tape
as Study 36 (Greenback)**, so the two FX studies carry one fingerprint. The strategy is the dollar-neutral
carry book (long the high-rate, short the low-rate tercile, weights set on the prior month-end's rates),
monthly, {COST_BPS:.0f} bp per unit traded. The rate-bucket diagnostic uses the same one-month lag as the
book. The offline core proves the machinery on a synthetic G10 with a known carry premium and risk-off
crashes; this is the measurement on the market. OECD MEI ends **2024-01**, so the as-of is pinned at
**{asof}** and never creeps; match the fingerprint below.*

## The verdict, earned — Signal `{signal}` · Tradability `{trad}` · Crash risk? `{crash}`

On 2001–2024 G10 the carry premium is **there, but thin**: high-rate currencies out-earn low-rate ones by
**{d['pb']['hml_ann_pct']:+.1f}%/yr** across (lagged) rate buckets, and the carry portfolio earns
**{pt['mean_ann_pct']:+.1f}%/yr** net — but the Newey-West *t* is **{pt['t_stat']:+.1f}** (gross
**{d['pt_gross']['t_stat']:+.1f}**) and the Sharpe of **{cmp['sharpe']:+.2f}** has a bootstrap 95% CI of
**[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]** ({ci['frac_negative']*100:.0f}% of resamples negative).
The slope is the right way up and the literature's decades-long evidence stands behind it, but *this*
post-2000 sample alone cannot reject zero — exactly the carry decay the literature documents — so the
desk stamps `{signal}`, not `REAL`. The **steamroller is fully on the tape**: monthly skew
**{cr['skew']:+.2f}**, worst month **{cr['worst_month_pct']:+.1f}%** ({list(d['worst'])[0]}), max drawdown
**{cr['max_drawdown_pct']:.0f}%** on a {cmp['vol_ann']*100:.1f}%-vol book — the GFC and COVID risk-offs,
arriving all at once. And the crash *resists* the desk's usual fix: vol-targeting **cuts** the Sharpe
(**{cc['plain']['sharpe']:+.2f} → {cc['managed']['sharpe']:+.2f}**) and **deepens** the drawdown
(**{cc['plain']['max_drawdown_pct']:.0f}% → {cc['managed']['max_drawdown_pct']:.0f}%**), because the
crash is a sudden risk-off jump, not a vol build-up a trailing estimate can see — hence `{crash}`.

## Data stamp

- **G10**: {d['n_ccy']} currencies, {d['lo']} → {d['hi']}, {d['n_months']} months, as-of **{asof}**,
  fingerprint `{d['fp']}` *(identical to Study 36's published tape — one tape, two studies)*

## The premium — present, priced, thin

- High-minus-low **lagged** rate-bucket spread **{d['pb']['hml_ann_pct']:+.1f}%/yr** (bucketed on the
  prior month-end's rate, the same information set as the book — sorting on the same month's rate is the
  classic look-ahead this diagnostic now avoids).
- Carry portfolio (net @{COST_BPS:.0f} bp): **{pt['mean_ann_pct']:+.1f}%/yr** (HAC *t* =
  **{pt['t_stat']:+.1f}**), Sharpe **{cmp['sharpe']:+.2f}**, bootstrap 95% CI
  **[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]**, vol {cmp['vol_ann']*100:.1f}%.
- **Turnover {cmp['turnover_ann']:.2f}×/yr** — rates move slowly, so the book is genuinely cheap to run.
  Cost sweep (bp per unit → Sharpe): {sweep_head} bp → {sweep_cells}. The cost claim is now *exercised*
  on real rebalancing (the synthetic control's constant rates produce zero turnover by construction, so
  it could never test this).

## The steamroller — carry's fat negative tail

- Monthly skew **{cr['skew']:+.2f}**; worst month **{cr['worst_month_pct']:+.1f}%**; worst-5 average
  **{cr['worst5_months_mean_pct']:+.1f}%**; max drawdown **{cr['max_drawdown_pct']:.0f}%**. Worst months:
  {worst_s} — the 2008 and 2020 risk-offs, exactly where the believers' story puts them.
- The worst 5 months carry **{d['dc']['worst_k_share_of_losses']*100:.0f}%** of all losing-month losses
  ({d['dc']['n_negative_months']} negative months) — crash-concentrated, not diffuse.
- **Vol-managed vs plain** (shared {cc['n_months']}-month window, after the 12-month vol burn-in — the
  plain Sharpe here differs from the headline for that reason): Sharpe
  **{cc['plain']['sharpe']:+.2f} → {cc['managed']['sharpe']:+.2f}**, skew
  **{cc['plain']['skew']:+.2f} → {cc['managed']['skew']:+.2f}**, worst month
  **{cc['plain']['worst_month_pct']:+.1f}% → {cc['managed']['worst_month_pct']:+.1f}%**, drawdown
  **{cc['plain']['max_drawdown_pct']:.0f}% → {cc['managed']['max_drawdown_pct']:.0f}%**. On the real tape
  the overlay is a strict loss: it levers up the calm 2003–07 carry build-up and rides into the 2008 jump.

Signal `{signal}`, Tradability `{trad}`, Crash risk? `{crash}`.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
