"""Real-data run — is the dollar-carry / carry⊕momentum combo real on G10, cache-first.

    python examples/verify.py            # cache-only (offline); prints if the FX/rates cache is present
    python examples/verify.py --fetch    # download short rates (FRED) + FX spot (yfinance), cache, run

Runs the carry, dollar-carry, momentum and combo books on the real tape and prints the carry premium, the
combo's Sharpe uplift over either leg, the leg correlation, and the carry crash — fingerprinted and
as-of pinned, writing ../docs/results.md.

**Pending-fetch note (Study 27 pattern).** This study's carry signal needs **short rates from FRED**,
whose download **times out in this sandbox**. FX spot from yfinance works, but without the rates there is
no carry signal — so with no pre-populated cache this prints a skip and the offline synthetic core
(run_synthetic_demo.py) is the validated proof. The real-tape verdict is PENDING one networked FRED fetch.
"""

from __future__ import annotations

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

from greenback import data, strategy, extension  # noqa: E402

COST_BPS = 10.0


def main(fetch: bool) -> None:
    tape = data.fetch_fx_rates(fetch=fetch)
    if not tape:
        print("[skip] no cached G10 FX/rates. This study's carry signal needs short rates from FRED,")
        print("       whose download is unavailable (times out) in this environment — see docs/results.md.")
        print("       Run `python examples/verify.py --fetch` where FRED is reachable to populate the cache.")
        print("       The offline synthetic core (run_synthetic_demo.py) is the validated proof meanwhile.")
        return

    rates, fx = tape["rates"], tape["fx"]
    try:
        from quantlab.repro import DEFAULT_AS_OF, as_of, fingerprint
        rates, fx = as_of(rates, DEFAULT_AS_OF), as_of(fx, DEFAULT_AS_OF)
    except Exception:
        DEFAULT_AS_OF, fingerprint = None, lambda d: "n/a"

    # build excess returns and rate differentials from the real tape
    import numpy as np
    import pandas as pd
    r = rates / 100.0 / 12.0
    rbar = r.mean(axis=1)
    fx_ret = np.log(fx).diff()
    cols = [c for c in fx.columns if c in r.columns]
    xr = pd.DataFrame({c: (r[c] - rbar) + fx_ret[c] for c in cols}).dropna(how="all")
    rate_diffs = (r[cols].sub(rbar, axis=0)).reindex(xr.index)

    carry = strategy.carry_returns(xr, rate_diffs, cost_bps=COST_BPS)
    mom = strategy.momentum_returns(xr, cost_bps=COST_BPS)
    combo = strategy.combine(carry, mom)
    d = extension.diversification(xr, rate_diffs, cost_bps=COST_BPS)
    cc = extension.carry_crash(xr, rate_diffs, cost_bps=COST_BPS)
    print(f"carry Sharpe {d['carry_sharpe']:+.2f}, momentum {d['momentum_sharpe']:+.2f}, combo {d['combo_sharpe']:+.2f}; "
          f"leg corr {d['leg_correlation']:+.2f}; carry skew {cc['carry_skew']:+.2f}, DD {cc['carry_max_dd_pct']:.0f}%")
    print(f"as-of {DEFAULT_AS_OF} · fingerprint {fingerprint(xr.round(6))}")
    print("(would write ../docs/results.md with the fingerprinted G10 numbers)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    main(ap.parse_args().fetch)
