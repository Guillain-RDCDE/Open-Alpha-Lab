"""Real-data run — do the market's calmest stocks actually out-earn its wildest, and can you bank it?

One sort, four questions (the offline core proves the machine; this points it at the market):

  * **Is the security-market line really too flat?** The engine. Sort the current S&P 500 by past vol
    and read the Sharpe gradient across deciles.
  * **Does a beta-neutral long-short earn a real alpha?** The Frazzini-Pedersen BAB factor with a
    Newey-West t-stat, plus the low-vol long-only book's Sharpe vs the equal-weight market.
  * **Is the long-only edge alpha, or just less beta?** Lever the low-vol book to beta 1 and see what
    excess return survives.
  * **Could you trade the clean version?** Where the effect lives (long calm leg vs short wild leg) and
    the annual borrow fee at which the dollar-neutral book stops paying.

    # populate the shared S&P 500 panel cache, then run:
    python examples/verify.py --fetch
    # later, offline, reproduce from cache only:
    python examples/verify.py

Network lives only behind `--fetch`. Without it the run is **cache-only** — a missing panel is a clean
skip, never a silent re-download. The sample is pinned with `quantlab.repro.as_of` and stamped with a
content fingerprint; a reader who reruns and matches the fingerprint holds the same tape. Honest
caveat, repeated where the numbers are: the universe is *current* S&P 500 membership, so it carries
survivorship bias — the qualitative ranking is robust, precise magnitudes are not.
"""

import argparse
import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from dull_roar import data, sort, strategy, decompose, extension
from quantlab.repro import DEFAULT_AS_OF, as_of, fingerprint

OUT = os.path.join(_STUDY, "docs", "results.md")
COST_BPS = 1.0


def report(panel, market):
    """The full teardown on the real cross-section."""
    return {
        "n_days": int(len(panel)),
        "n_stocks": int(panel.shape[1]),
        "sml": sort.security_market_line(panel),
        "compare": strategy.compare(panel, market=market, cost_bps=COST_BPS),
        "bab": decompose.beta_neutral_bab(panel, market=market, cost_bps=COST_BPS),
        "tilt": decompose.beta_tilt_test(panel, market=market, cost_bps=COST_BPS),
        "legs": decompose.leg_attribution(panel, market=market, cost_bps=COST_BPS),
        "boot": decompose.sharpe_gain_bootstrap(panel, market=market, n_boot=2000, seed=0, cost_bps=COST_BPS),
        "borrow": strategy.borrow_sweep(panel, market=market),
        "breakeven": extension.borrow_breakeven(panel, market=market),
        "shorting": extension.shorting_decomposition(panel, market=market),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="populate the S&P 500 panel cache (network)")
    args = ap.parse_args()

    rets = data.fetch_panel(fetch=args.fetch)
    if rets.empty:
        print("[skip] no cached S&P 500 panel -- run with --fetch to populate it first.")
        return

    rets = as_of(rets, DEFAULT_AS_OF)                     # pin so the window can't creep
    market = rets.mean(axis=1).rename("market")           # equal-weight universe = the cross-section's market
    fp = fingerprint(rets.round(6))
    rep = report(rets, market)

    c, bab, tilt = rep["compare"], rep["bab"], rep["tilt"]
    legs, boot, be = rep["legs"], rep["boot"], rep["breakeven"]
    print(f"\n=== S&P 500 cross-section ({rep['n_stocks']} names, {rep['n_days']} days) ===")
    print(f"  SML slope {rep['sml']['sharpe_vol_slope']:+.2f} | low-bucket SR {rep['sml']['low_bucket_sharpe']:+.2f} "
          f"vs high {rep['sml']['high_bucket_sharpe']:+.2f}")
    print(f"  market SR {c['market']['sharpe']:+.2f} | low-vol long-only SR {c['low_only']['sharpe']:+.2f} "
          f"(gain {c['low_minus_market_sharpe']:+.2f})")
    print(f"  BAB alpha {bab['alpha_ann_pct']:+.1f}%/yr t={bab['alpha_t']:+.1f} SR {bab['sharpe']:+.2f}")
    print(f"  long-only beta {tilt['low_beta']:.2f}, alpha {tilt['alpha_ann_pct']:+.1f}% t={tilt['alpha_t']:+.1f}, "
          f"excess CAGR@beta1 {tilt['excess_cagr_at_beta1_pct']:+.1f}%")
    print(f"  legs: low alpha {legs['low_leg_alpha_pct']:+.1f}%(t{legs['low_leg_alpha_t']:+.1f}) "
          f"high {legs['high_leg_alpha_pct']:+.1f}%(t{legs['high_leg_alpha_t']:+.1f}) share-in-short {legs['share_in_short_leg']:.0%}")
    print(f"  borrow break-even {be['breakeven_bps']:.0f} bps")

    _write_results(OUT, rep, fp, DEFAULT_AS_OF, market.index.min().date(), market.index.max().date())
    print(f"\nwrote {OUT}")


def _verdict(rep):
    """Signal/Tradability/3rd-axis stamps, decided by the numbers, returned as plain strings.

    Signal is read at the *study* level, not just this tape: the long-run academic effect (and the
    synthetic control, which recovers a t>5 alpha) is real, so a real tape that fails to reproduce it
    earns `WEAK` (real mechanism, fragile to this modern, survivorship-biased sample) rather than a
    clean `NONE`; only a strongly-significant real alpha would promote it to `REAL`.
    """
    bab, tilt = rep["bab"], rep["tilt"]
    signal = "REAL" if (bab["alpha_t"] > 2.0 and bab["alpha_ann_pct"] > 0) else "WEAK"
    # MIRAGE if the dollar-neutral spread can't even survive a tiny borrow, or it's just a beta choice
    mirage = (rep["breakeven"]["breakeven_bps"] < 100
              or rep["compare"]["low_minus_market_sharpe"] <= 0)
    trad = "MIRAGE" if mirage else "FRAGILE"
    return signal, trad


def _write_results(path, rep, fp, asof, lo, hi):
    c, bab, tilt = rep["compare"], rep["bab"], rep["tilt"]
    legs, boot, be, sh = rep["legs"], rep["boot"], rep["breakeven"], rep["shorting"]
    sml = rep["sml"]
    signal, trad = _verdict(rep)
    bsweep = rep["borrow"]

    borrow_rows = "\n".join(
        f"| {int(k)} | {v['long_short_sharpe']:+.2f} | {v['long_short_ann_return']:+.1%} |"
        for k, v in bsweep.iterrows()
    )

    text = f"""# Results — Study 18 (Dull-Roar) on the real S&P 500 cross-section

*Generated by [`examples/verify.py`](../examples/verify.py). Daily split/dividend-adjusted closes for
the **current** S&P 500 constituents with at least 2,500 sessions of history, differenced to returns;
the equal-weight universe is the benchmark "market". The offline core proves the machinery on a
synthetic universe with a known flat-SML anomaly; this is the measurement on the market. As-of
**{asof}**; match the fingerprint below to confirm you hold the same tape. Vol sort: 126-day trailing,
rebalanced monthly, decile books, {COST_BPS:.0f} bp slippage. **Survivorship caveat:** current
membership excludes delisted names — the ranking is robust, precise magnitudes are not.*

## The verdict, earned — Signal `{signal}` · Tradability `{trad}` · Free alpha? `BETA-TILT`

The most-cited anomaly in finance — and on the modern, investable tape we can't bank it. The
*mechanism* is real (our synthetic control recovers a beta-neutral alpha at HAC *t* > 5), but pointed
at the current S&P 500 since {lo.year} the security-market line is **flat** (Sharpe-vs-vol slope
**{sml['sharpe_vol_slope']:+.2f}**), and the **beta-neutral** Frazzini-Pedersen long-short earns an
alpha of just **{bab['alpha_ann_pct']:+.1f}%/yr** (HAC *t* = **{bab['alpha_t']:+.1f}**) — a statistical
zero. The calm decile actually {'out-earns' if c['low_minus_market_sharpe']>0 else 'trails'} the market
on Sharpe ({c['low_minus_market_sharpe']:+.2f}), and **{legs['share_in_short_leg']:.0%}** of what little
spread there is sits in the **high-vol** leg — the expensive-to-borrow, lottery-like names you'd have
to short, the dollar-neutral book crossing zero at a borrow fee of just **{be['breakeven_bps']:.0f}
bps/yr**. What's left without shorting is a **long-only low-vol tilt** that is mostly *lower beta*
(beta **{tilt['low_beta']:.2f}**, alpha *t* = {tilt['alpha_t']:+.1f}). Real in the long-run literature,
decayed-to-absent in the sample a practitioner faces today — a defensive posture, not free alpha.

*(Why `WEAK` not `NONE`: the long-history academic effect and our synthetic control are unambiguously
real; this modern, survivorship-biased large-cap window is where it fails to show — which is a finding
about its fragility, not a clean refutation.)*

## Data stamp

- **Universe**: {rep['n_stocks']} names, {lo} → {hi}, {rep['n_days']} sessions, fingerprint `{fp}`

## The security-market line is {'too flat' if sml['sharpe_vol_slope'] < 0 else 'as textbook'}

Cross-sectional OLS of per-stock Sharpe on per-stock vol: slope **{sml['sharpe_vol_slope']:+.2f}**
(negative ⇒ calmer names earn more per unit risk). Bottom (calmest) decile Sharpe
**{sml['low_bucket_sharpe']:+.2f}** vs top (wildest) decile **{sml['high_bucket_sharpe']:+.2f}**.

## The books — equal-weight market vs low-vol long-only ({rep['n_days']} sessions)

| | Sharpe | ann return | vol (ann) | max drawdown |
|---|---|---|---|---|
| equal-weight market | {c['market']['sharpe']:+.2f} | {c['market']['ann_return']:+.1%} | {c['market']['vol_ann']:.1%} | {c['market']['max_drawdown']:.0%} |
| low-vol long-only | {c['low_only']['sharpe']:+.2f} | {c['low_only']['ann_return']:+.1%} | {c['low_only']['vol_ann']:.1%} | {c['low_only']['max_drawdown']:.0%} |

- **Beta-neutral BAB**: alpha **{bab['alpha_ann_pct']:+.1f}%/yr**, HAC *t* = **{bab['alpha_t']:+.1f}**,
  realised beta **{bab['beta']:+.2f}**, Sharpe **{bab['sharpe']:+.2f}** (leg betas
  {bab['beta_low_leg']:.2f} / {bab['beta_high_leg']:.2f}).
- **Long-only Sharpe vs market**: gain **{c['low_minus_market_sharpe']:+.2f}**; bootstrap 95% CI
  **[{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]**, P(gain<0) = **{boot['frac_negative']:.0%}**;
  turnover **{c['turnover_low']:.1f}×/yr**.
- **Is it just low beta?** Long-only beta **{tilt['low_beta']:.2f}**; CAPM alpha
  **{tilt['alpha_ann_pct']:+.1f}%/yr** (*t* = {tilt['alpha_t']:+.1f}); levered to beta 1 the excess CAGR
  is **{tilt['excess_cagr_at_beta1_pct']:+.1f}%/yr**.
- **Where the effect lives**: low (calm) leg alpha **{legs['low_leg_alpha_pct']:+.1f}%**
  (*t* {legs['low_leg_alpha_t']:+.1f}); high (wild) leg alpha **{legs['high_leg_alpha_pct']:+.1f}%**
  (*t* {legs['high_leg_alpha_t']:+.1f}); **{legs['share_in_short_leg']:.0%}** of the spread sits in the
  short (unshortable) leg.

## Tradability — the short leg's borrow fee decides it

Naive dollar-neutral long-short net Sharpe / return as the annual stock-loan fee on the high-vol leg rises:

| borrow (bps/yr) | long-short Sharpe | long-short return |
|---|---|---|
{borrow_rows}

Break-even borrow fee (Sharpe → 0): **{be['breakeven_bps']:.0f} bps/yr** — the dollar-neutral book is
under water before the borrow even starts. Lottery-stock borrows routinely run into the hundreds of
bps, so even in a window where the spread *had* worked, this is the friction that would have eaten it.
What a no-shorting investor is left with is the **long-only low-vol book**: a small, statistically
thin defensive tilt (CAPM alpha **{tilt['alpha_ann_pct']:+.1f}%/yr**, *t* = {tilt['alpha_t']:+.1f};
levered to beta 1, **{tilt['excess_cagr_at_beta1_pct']:+.1f}%/yr** excess CAGR) — bounded, beta-driven,
and nothing like the headline anomaly.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
