"""Reproduce the real-data headline run (docs/results.md) — S&P 500, 1950–today.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download ^GSPC + ^IRX from Yahoo, then run
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import DEFAULT_AS_OF, as_of  # noqa: E402

from hangover import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_index(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network) to download ^GSPC.")
        return
    ret = as_of(ret.to_frame(), DEFAULT_AS_OF)["ret"]  # pin the sample; drops any in-progress month
    tbill = data.fetch_tbill(fetch=fetch)
    if not tbill.empty:
        tbill = as_of(tbill.to_frame(), DEFAULT_AS_OF)["tbill"]
    tbl = st.annual_table(ret)
    print(f"\nS&P 500 January Barometer, {tbl.index.min()}-{tbl.index.max()} ({len(tbl)} years)")
    print("(^GSPC is price-only — no dividends; cash years credited the ^IRX T-bill, zero pre-1960)\n")
    cm = st.conditional_means(tbl)
    print(cm.round(4).to_string())
    sp = st.split_tests(tbl)
    print(f"\n  up vs down January P(rest-of-year up): {sp['k_up']}/{sp['n_up']} vs {sp['k_down']}/{sp['n_down']}"
          f"  -> Fisher exact p = {sp['fisher_p']:.4f}")
    print(f"  rest-of-year mean gap {sp['mean_gap']:+.2%}  -> permutation p = {sp['mean_perm_p']:.4f} (seeded, 20k shuffles)")

    ac = st.accuracy_with_ci(tbl)
    print(f"\nDirectional accuracy: {ac['accuracy']:.1%} [Wilson {ac['acc_low']:.1%}, {ac['acc_high']:.1%}]"
          f"  vs  always-predict-up base rate: {ac['base_rate']:.1%} [Wilson {ac['base_low']:.1%}, {ac['base_high']:.1%}]")

    cash = st.cash_roy(tbill) if not tbill.empty else None
    bar, bh = st.barometer_returns(tbl, cash), st.buy_hold_roy(tbl)
    for nm, s in [("Buy & hold (Feb-Dec)", bh), ("Barometer (T-bill if Jan down)", bar)]:
        d = st.summary(s)
        print(f"  {nm:31} CAGR {d['cagr']:+.2%}  Sharpe {d['sharpe']:.2f}  maxDD {d['max_drawdown']:.1%}")
    print("  (price-only race: a total-return tape would add ~3-4%/yr of dividends to every invested year")
    print("   — all of buy-and-hold's, but only the barometer's ~60% invested years, so this race flatters the omen)")

    dc = st.decay_split(tbl)
    for lab, era in [("1950-1972", "pre"), ("1973-on", "post")]:
        print(f"  decay {lab}: accuracy {dc[era + '_acc']:.1%} ({dc[era + '_k']}/{dc[era + '_n']})"
              f" [Wilson {dc[era + '_low']:.1%}, {dc[era + '_high']:.1%}],"
              f" jan-down rest-of-year mean {dc[era + '_down_mean']:+.2%}")
    print(f"  pre-vs-post accuracy difference: Fisher exact p = {dc['fisher_p']:.4f}"
          f"  ({'not distinguishable from noise' if dc['fisher_p'] > 0.05 else 'significant at 5%'})")

    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(ret.to_frame())}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
