"""Real-data run — is macro momentum a premium on real cross-asset ETFs, and does the inflation hedge pay?

    python examples/verify.py            # cache-only (offline) — the headline run
    python examples/verify.py --fetch    # accepted for parity; the tape is pre-cached, so this is identical

The real tape is the desk's three cached parquets — ``macro_us`` (BLS CPI level, industrial production),
``us_treasury_yields`` (the y10y−y3m slope, a daily growth proxy), and ``cross_asset_etfs`` (18 liquid
ETFs). All macro drivers are lagged one month for publication delay; the sample begins ~2007 (when the
real-asset ETFs that make the inflation hedge tradable exist). Offline; fingerprinted via ``quantlab.repro``.
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

import numpy as np

from barometer import costs, data, extension, strategy
from barometer.data import REAL_ASSET_ORDER, REAL_ASSETS

COST_BPS = 5.0
SPEC = dict(assets=REAL_ASSETS, order=REAL_ASSET_ORDER)


def newey_west_t(x, lags: int = 6) -> float:
    """HAC (Newey-West) t-stat for the mean of a monthly return series (lags ≈ √n, here 6)."""
    x = np.asarray([v for v in x if v == v], dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = (e[k:] @ e[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()

    bundle = data.fetch_macro(fetch=args.fetch)
    if not bundle:
        print("[skip] cache parquets (macro_us / us_treasury_yields / cross_asset_etfs) not found in _cache.")
        return

    macro, rets = bundle["macro"], bundle["assets"]
    as_of_str = fp = "n/a"
    try:
        from quantlab.repro import DEFAULT_AS_OF, as_of, fingerprint
        macro = as_of(macro, DEFAULT_AS_OF)
        rets = rets.loc[macro.index]
        fp = fingerprint(rets.round(6))
        as_of_str = str(rets.index.max().date())
    except Exception:
        as_of_str = str(rets.index.max().date())

    lo, hi = rets.index.min().date(), rets.index.max().date()
    print(f"[data] real cross-asset book: {len(rets)} months  {lo} -> {hi}  as-of {as_of_str}  fingerprint {fp}")
    print(f"       {rets.shape[1]} ETFs; macro = CPI-YoY (inflation) + yield-curve slope (growth), lagged 1m\n")

    cmp = strategy.compare(rets, macro, cost_bps=COST_BPS, **SPEC)
    for kind in ("macro_momentum", "inflation"):
        s = cmp[kind]
        r = strategy.book_returns(rets, macro, kind=kind, cost_bps=COST_BPS, **SPEC)
        t = newey_west_t(r)
        print(f"  {kind:15} Sharpe {s['sharpe']:+.2f}  CAGR {100*s['cagr']:+.1f}%  maxDD {100*s['max_drawdown']:.0f}%  "
              f"turnover {s['turnover_ann']:.1f}x/yr  HAC t {t:+.2f}")

    print(f"\n  break-even  macro-momentum {costs.breakeven_cost_bps(rets, macro, 'macro_momentum', **SPEC):.0f} bp"
          f"   inflation {costs.breakeven_cost_bps(rets, macro, 'inflation', **SPEC):.0f} bp")

    print("\ncost sweep (macro-momentum, net Sharpe):")
    print(costs.cost_sweep(rets, macro, kind="macro_momentum", **SPEC).round(3).to_string())

    print("\nregime split — inflation-hedge book (rising vs falling inflation):")
    print(extension.regime_split(rets, macro, kind="inflation", cost_bps=COST_BPS, **SPEC).round(2).to_string())
    print("\nregime split — RAW real-minus-nominal spread (always long, timing stripped out):")
    print(extension.raw_real_minus_nominal(rets, macro, **SPEC).round(2).to_string())

    # Benchmarks: equal-weight the 18 ETFs, and a 60/40 SPY/IEF hold.
    ew = rets[REAL_ASSET_ORDER].mean(axis=1)
    sixty = 0.6 * rets["SPY"] + 0.4 * rets["IEF"]
    print("\nbenchmarks (buy-and-hold of the same assets):")
    for nm, b in (("equal-weight 18", ew), ("60/40 SPY/IEF", sixty)):
        s = strategy.summary(b)
        print(f"  {nm:16} Sharpe {s['sharpe']:+.2f}  CAGR {100*s['cagr']:+.1f}%  maxDD {100*s['max_drawdown']:.0f}%")


if __name__ == "__main__":
    main()
