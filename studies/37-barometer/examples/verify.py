"""Real-data run (PENDING) — is macro momentum a premium on real assets, and does the inflation hedge pay?

    python examples/verify.py            # cache-only (offline); prints the cache-miss message
    python examples/verify.py --fetch    # try to pull FRED macro + asset proxies, then run

**Honest network note (Steamroller pattern).** This study's real tape is **FRED macro series** (growth:
INDPRO/PAYEMS; inflation: CPIAUCSL/T10YIE) plus liquid asset proxies. In this sandbox the daily FRED
series (T10YIE, DGS10) reliably time out, and even the monthly CPI series succeeds only intermittently, so
the full macro+asset panel is **not reliably fetchable**. On a cache-miss this prints the message below and
the synthetic core (run_synthetic_demo.py) is the validated proof. The verdict in docs/results.md is earned
on that control plus the literature; the real run is **pending a reliable FRED macro fetch** as of
2026-06-10. When a cache is present this script fingerprints the run with `quantlab.repro`.
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

from barometer import costs, data, extension, strategy

COST_BPS = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()

    bundle = data.fetch_macro(fetch=args.fetch)
    if not bundle:
        print("[skip] FRED macro series unavailable/unreliable in this environment — see docs/results.md")
        print("       The macro state needs FRED growth/inflation series (INDPRO/PAYEMS/CPIAUCSL/T10YIE);")
        print("       the daily ones time out here and monthly CPI is intermittent, so the real run is")
        print("       PENDING a reliable FRED macro fetch (as of 2026-06-10).")
        print("       The offline synthetic core (examples/run_synthetic_demo.py) is the validated proof.")
        return

    macro, assets = bundle["macro"], bundle["assets"]
    try:
        from quantlab.repro import DEFAULT_AS_OF, as_of, fingerprint
        macro = as_of(macro, DEFAULT_AS_OF)
        assets = as_of(assets, DEFAULT_AS_OF)
        fp = fingerprint(assets.round(6))
    except Exception:
        fp = "n/a"

    rets = assets.pct_change().dropna(how="all")
    cmp = strategy.compare(rets, macro, cost_bps=COST_BPS)
    print(f"macro-momentum Sharpe {cmp['macro_momentum']['sharpe']:+.2f}; "
          f"inflation-hedge Sharpe {cmp['inflation']['sharpe']:+.2f}; fingerprint {fp}")
    print("regime split (inflation book):")
    print(extension.regime_split(rets, macro, kind="inflation", cost_bps=COST_BPS).round(2).to_string())
    print(f"break-even macro-momentum: {costs.breakeven_cost_bps(rets, macro, 'macro_momentum'):.0f} bp")


if __name__ == "__main__":
    main()
