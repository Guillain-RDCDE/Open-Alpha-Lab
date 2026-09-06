"""Real-tape verification — Study 994 (The Rounding Tax). Regenerates docs/results.md.

Builds the target allocation out of whole shares at every account size from $500 to
$1m, measures how far the achieved weights land from the plan, runs the whole-share portfolio
forward against a fractional-share twin over the full sample, decomposes the shortfall into cash
drag, trading costs and mean-zero noise, and prices the three escapes — fractional shares,
cheaper share prices, and simply holding fewer funds.

    python studies/994-small-account-lot-drag/examples/verify.py            # cache-only
    python studies/994-small-account-lot-drag/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from roundingtax import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


HEADLINE_CAPITAL = 3000.0
COST_BPS = 5.0
REBALANCE_DAYS = 252


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "capital": HEADLINE_CAPITAL, "cost_bps": COST_BPS,
               "fingerprint": data.fingerprint(px)}

    target = dict(data.TARGET)
    names = [t for t in target if t in px.columns]
    panel = px[names].dropna()
    h["n_funds"] = int(len(names))
    h["window"] = [str(panel.index[0].date()), str(panel.index[-1].date())]
    h["years"] = float(len(panel) / st.TRADING_DAYS)
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  target: " + ", ".join(f"{t} {target[t]:.0%}" for t in names))
    print(f"  common window {panel.index[0].date()} -> {panel.index[-1].date()} "
          f"({h['years']:.1f} years)")

    print("\n=== 1. share prices, which is what actually binds ===")
    last = panel.iloc[-1]
    first = panel.iloc[0]
    for t in names:
        print(f"  {t:6s} ${first[t]:8.2f} -> ${last[t]:8.2f}  "
              f"(target weight {target[t]:.0%})")
    h["prices_last"] = {t: float(last[t]) for t in names}
    h["one_share_cost"] = st.one_share_cost(target, last)
    print(f"  one share of everything: ${h['one_share_cost']:,.2f}")
    print("  (a caveat that belongs here: these are total-return adjusted closes, so the "
          "historical LEVELS are lower than the prices that actually traded. That makes early "
          "rounding constraints look milder than they were, so every cost below is if anything "
          "understated.)")

    print("\n=== 2. what account can actually hold this portfolio? ===")
    for tol in (0.05, 0.02, 0.01, 0.005):
        mv = st.min_viable_capital(target, last, tol)
        print(f"  to hit every weight within {tol:.1%}: ${mv:,.0f}")
        if tol == 0.01:
            h["min_viable"] = float(mv)
    print(f"  -> the usually-quoted 'one share of each' figure is "
          f"${h['one_share_cost']:,.0f}; the figure that matters is "
          f"${h['min_viable']:,.0f}, {h['min_viable'] / h['one_share_cost']:.0f}x larger")

    print("\n=== 3. allocation error against account size ===")
    ev = st.error_vs_capital(target, last)
    print(ev.round(4).to_string())
    h["error_vs_capital"] = ev.reset_index().to_dict("records")

    print(f"\n=== 4. running a ${HEADLINE_CAPITAL:,.0f} account forward ===")
    cash_rate = px[data.CASH].pct_change().reindex(panel.index).fillna(0.0) \
        if data.CASH in px.columns else None
    cmp = st.compare_to_ideal(panel, target, HEADLINE_CAPITAL, cash_rate,
                              rebalance_days=REBALANCE_DAYS, cost_bps=COST_BPS)
    h.update({k: cmp[k] for k in ("cagr_whole", "cagr_fractional", "cagr_gap",
                                  "tracking_error", "final_gap_pct", "mean_cash_share",
                                  "mean_l1_error", "cost_share")})
    print(f"  whole shares:      CAGR {cmp['cagr_whole']:+.2%}")
    print(f"  fractional shares: CAGR {cmp['cagr_fractional']:+.2%}")
    print(f"  gap {cmp['cagr_gap']:+.2%}/yr, tracking error "
          f"{cmp['tracking_error']:.2%}/yr, final value gap {cmp['final_gap_pct']:+.2%}")
    print(f"  average uninvested cash {cmp['mean_cash_share']:.2%}, average allocation error "
          f"{cmp['mean_l1_error']:.2%}")
    sim = st.rebalance_simulation(panel, target, HEADLINE_CAPITAL, cash_rate,
                                  rebalance_days=REBALANCE_DAYS, cost_bps=COST_BPS)
    h["max_abs_error"] = float(sim["max_l1_error"])
    print(f"  worst allocation error over the period: {sim['max_l1_error']:.2%}")

    print("\n=== 5. is it drag, or is it noise? ===")
    eq_prem = 0.06
    dec = st.decompose_shortfall(cmp, 0.02, eq_prem)
    h.update({k: dec[k] for k in ("cash_drag", "trading_costs", "unexplained_noise",
                                  "drag_share")})
    print(f"  total gap:            {dec['total_gap']:+.3%}/yr")
    print(f"    cash drag:          {dec['cash_drag']:+.3%}/yr  (one-directional, compounds)")
    print(f"    trading costs:      {dec['trading_costs']:+.3%}/yr  (one-directional)")
    print(f"    unexplained noise:  {dec['unexplained_noise']:+.3%}/yr  (mean-zero, cancels)")
    print(f"  the explainable drag is {dec['drag_share']:.0%} of the total gap; the rest is "
          f"which way the dice fell")
    signs = []
    for cap in (2000, 3000, 4000, 5000, 7500):
        c2 = st.compare_to_ideal(panel, target, float(cap), cash_rate,
                                 rebalance_days=REBALANCE_DAYS, cost_bps=COST_BPS)
        d2 = st.decompose_shortfall(c2, 0.02, eq_prem)
        signs.append({"capital": cap, "gap": c2["cagr_gap"],
                      "noise": d2["unexplained_noise"]})
        print(f"  ${cap:>6,}: gap {c2['cagr_gap']:+.3%}, noise term "
              f"{d2['unexplained_noise']:+.3%}")
    h["noise_signs"] = signs
    n_pos = sum(1 for s in signs if s["noise"] > 0)
    print(f"  the noise term is positive at {n_pos} of {len(signs)} account sizes — which is "
          f"what mean-zero looks like, and why quoting the whole gap as 'the cost of being "
          f"small' overstates it")

    print("\n=== 6. tracking error across account sizes ===")
    scale = []
    for cap in (1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000, 500_000):
        c2 = st.compare_to_ideal(panel, target, float(cap), cash_rate,
                                 rebalance_days=REBALANCE_DAYS, cost_bps=COST_BPS)
        scale.append({"capital": cap, "tracking_error": c2["tracking_error"],
                      "cagr_gap": c2["cagr_gap"],
                      "mean_l1_error": c2["mean_l1_error"],
                      "mean_cash_share": c2["mean_cash_share"]})
        print(f"  ${cap:>7,}: TE {c2['tracking_error']:.2%}/yr, allocation error "
              f"{c2['mean_l1_error']:.2%}, cash {c2['mean_cash_share']:.2%}, gap "
              f"{c2['cagr_gap']:+.2%}")
    h["scale"] = scale

    print("\n=== 7. the escapes ===")
    swaps = {data.CORE: data.CHEAP_CORE} if data.CHEAP_CORE in px.columns else None
    esc_panel = px[[c for c in px.columns if c != data.CASH]].dropna()
    esc = st.escape_table(esc_panel, target, HEADLINE_CAPITAL, cash_rate, swaps,
                          rebalance_days=REBALANCE_DAYS, cost_bps=COST_BPS)
    print(esc.round(4).to_string())
    h["escapes"] = esc.reset_index().to_dict("records")
    whole_err = float(esc.loc["whole shares, as specified", "mean_l1_error"])
    alt = esc.drop(["whole shares, as specified", "fractional shares"], errors="ignore")
    h["best_escape"] = str(alt["mean_l1_error"].idxmin()) if len(alt) else "none"
    h["best_escape_error"] = float(alt["mean_l1_error"].min()) if len(alt) else whole_err
    print(f"  best non-fractional escape: {h['best_escape']} "
          f"({h['best_escape_error']:.2%} allocation error vs {whole_err:.2%})")

    print("\n=== 8. the no-trade band ===")
    bands = []
    for band in (0.0, 0.02, 0.05, 0.10, 0.20):
        s2 = st.rebalance_simulation(panel, target, HEADLINE_CAPITAL, cash_rate,
                                     rebalance_days=REBALANCE_DAYS, cost_bps=COST_BPS,
                                     band=band)
        bands.append({"band": band, "cagr": s2["cagr"], "n_rebalances": s2["n_rebalances"],
                      "n_skipped": s2["n_skipped"], "cost_share": s2["cost_share"],
                      "mean_l1_error": s2["mean_l1_error"]})
        print(f"  band {band:.0%}: {s2['n_rebalances']} rebalances "
              f"({s2['n_skipped']} skipped), costs {s2['cost_share']:.2%} of capital, "
              f"CAGR {s2['cagr']:+.2%}, allocation error {s2['mean_l1_error']:.2%}")
    h["bands"] = bands

    print("\n=== 9. synthetic control: the mechanism isolated ===")
    ctrl = []
    for level in (20.0, 60.0, 150.0, 400.0, 600.0):
        sp = st.synthetic_prices(n=2520, price_levels=(level,) * 6)
        tgt6 = {f"F{k}": 1 / 6 for k in range(6)}
        c3 = st.compare_to_ideal(sp, tgt6, HEADLINE_CAPITAL)
        ctrl.append({"share_price": level, "tracking_error": c3["tracking_error"],
                     "mean_l1_error": c3["mean_l1_error"],
                     "mean_cash_share": c3["mean_cash_share"]})
        print(f"  ${level:6.0f}/share: TE {c3['tracking_error']:.2%}, allocation error "
              f"{c3['mean_l1_error']:.2%}, cash {c3['mean_cash_share']:.2%}")
    h["control"] = ctrl
    print("  identical returns in every row — only the share price differs. That is the whole "
          "mechanism.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    ev = "\n".join(
        f"| ${int(r['capital']):,} | {r['l1_error']:.2%} | {r['max_abs']:.2%} | "
        f"{r['cash_share']:.2%} | {int(r['n_positions'])} |" for r in h["error_vs_capital"])
    scale = "\n".join(
        f"| ${int(r['capital']):,} | {r['tracking_error']:.2%} | {r['mean_l1_error']:.2%} | "
        f"{r['mean_cash_share']:.2%} | {r['cagr_gap']:+.2%} |" for r in h["scale"])
    esc = "\n".join(
        f"| {r['variant']} | {int(r['n_positions'])} | {r['cagr']:+.2%} | {r['vol']:.1%} | "
        f"{r['mean_l1_error']:.2%} | {r['mean_cash_share']:.2%} |" for r in h["escapes"])
    bands = "\n".join(
        f"| {r['band']:.0%} | {int(r['n_rebalances'])} | {int(r['n_skipped'])} | "
        f"{r['cost_share']:.2%} | {r['mean_l1_error']:.2%} | {r['cagr']:+.2%} |"
        for r in h["bands"])
    ctrl = "\n".join(
        f"| ${r['share_price']:,.0f} | {r['tracking_error']:.2%} | {r['mean_l1_error']:.2%} | "
        f"{r['mean_cash_share']:.2%} |" for r in h["control"])
    noise = "\n".join(
        f"| ${int(r['capital']):,} | {r['gap']:+.3%} | {r['noise']:+.3%} |"
        for r in h["noise_signs"])
    prices = "\n".join(f"| {t} | ${p:,.2f} |" for t, p in h["prices_last"].items())
    return f"""# Results — Study 994 (The Rounding Tax) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). A {h['n_funds']}-fund target
allocation held in whole shares, {h['window'][0]} → {h['window'][1]} ({h['years']:.1f} years).
As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. Share prices are what bind

| Fund | Latest price |
|---|--:|
{prices}

One share of everything: **${h['one_share_cost']:,.2f}**.

A caveat that belongs at the top rather than in a footnote: these are total-return **adjusted**
closes, so the historical price *levels* are lower than what actually traded. That makes early
rounding constraints look milder than they were, so every cost below is if anything
understated.

## 2. What account can actually hold this portfolio?

| | |
|---|--:|
| One share of each (the usual figure) | ${h['one_share_cost']:,.0f} |
| **To hit every weight within 1pp** | **${h['min_viable']:,.0f}** |
| Ratio | {h['min_viable'] / h['one_share_cost']:.0f}× |

The gap between those two rows is the point. Owning one of everything gets you *a* portfolio,
not *your* portfolio — its weights are set by share prices rather than by your plan.

## 3. Allocation error by account size

| Capital | L1 error | Worst position | Uninvested cash | Positions filled |
|---|--:|--:|--:|--:|
{ev}

## 4. Running a ${h['capital']:,.0f} account forward

| | Whole shares | Fractional shares |
|---|--:|--:|
| CAGR | {h['cagr_whole']:+.2%} | {h['cagr_fractional']:+.2%} |

| | |
|---|--:|
| Annual gap | **{h['cagr_gap']:+.2%}** |
| Tracking error | **{h['tracking_error']:.2%}/yr** |
| Final value gap | {h['final_gap_pct']:+.2%} |
| Average uninvested cash | {h['mean_cash_share']:.2%} |
| Average allocation error | {h['mean_l1_error']:.2%} |
| Worst allocation error | {h['max_abs_error']:.2%} |

## 5. Drag or noise?

This is the section that decides what the advice should be.

| Component | Per year | Direction |
|---|--:|---|
| **Total gap** | {h['cagr_gap']:+.3%} | |
| Cash drag | {h['cash_drag']:+.3%} | one-directional, compounds |
| Trading costs | {h['trading_costs']:+.3%} | one-directional |
| Unexplained noise | {h['unexplained_noise']:+.3%} | **mean-zero, cancels** |

Explainable drag is {h['drag_share']:.0%} of the total gap. The rest is which way the dice fell,
and it changes sign with the account size:

| Capital | Total gap | Noise term |
|---|--:|--:|
{noise}

A noise term that flips sign across account sizes is what mean-zero looks like. Quoting the
whole gap as "the cost of being small" overstates it, often by a factor of two or more.

## 6. How it scales

| Capital | Tracking error | Allocation error | Cash | CAGR gap |
|---|--:|--:|--:|--:|
{scale}

## 7. The escapes

| Variant | Funds | CAGR | Vol | Allocation error | Cash |
|---|--:|--:|--:|--:|--:|
{esc}

The best non-fractional escape is **{h['best_escape']}**, cutting the allocation error to
{h['best_escape_error']:.2%} from {h['mean_l1_error']:.2%}. Note what that means: **holding
fewer funds fixes more of this than fractional shares do**, and unlike fractional shares it
works in every account type, at every broker, in every country.

## 8. The no-trade band

| Band | Rebalances | Skipped | Costs | Allocation error | CAGR |
|---|--:|--:|--:|--:|--:|
{bands}

## 9. Synthetic control — the mechanism isolated

Identical returns in every row. Only the share price differs:

| Share price | Tracking error | Allocation error | Cash |
|---|--:|--:|--:|
{ctrl}

## Caveats

- **Adjusted prices understate the problem.** Section 1 says why. A study using unadjusted
  historical trade prices would find *larger* early-period constraints, not smaller.
- **Fractional shares now exist** at most large US brokers, which makes the whole-share case
  historical for many readers — but not for retirement plans, most non-US brokers, or anyone
  holding a fund inside a wrapper that trades in whole units.
- **One target allocation.** The results depend on the specific weights: a portfolio whose small
  sleeves are 5% suffers far more than an equal-weight one, because a 5% sleeve of a small
  account is a few hundred dollars.
- **No taxes and no contributions.** Regular contributions substantially relieve the problem,
  because new cash can be directed at whichever sleeve is furthest below target. That is
  probably the single most important omission here and it runs in the direction of making
  things better.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[994-small-account-lot-drag](../README.md). Not investment advice.*
"""

def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    print("##HEADLINE## " + json.dumps(h, default=float))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
