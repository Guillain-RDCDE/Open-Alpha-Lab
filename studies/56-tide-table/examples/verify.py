"""Reproduce the real-data headline run (docs/results.md) — CAPE vs forward returns.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download Shiller's series (datahub mirror), then run
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

from tide_table import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_shiller(fetch=fetch, start_year=1900)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    ten = st.forward_corr(d["cape"], d["fwd10"])
    one = st.forward_corr(d["cape"], d["fwd1"])
    print(f"\nCAPE → forward real returns, {d.index.min().date()}..{d.index.max().date()}\n")
    print(f"  10-year horizon: corr(CAPE, fwd) {ten['corr_cape']:+.2f}, R²={ten['r2']:.2f} (n={ten['n']})")
    print(f"   1-year horizon: corr(CAPE, fwd) {one['corr_cape']:+.2f}, R²={one['r2']:.2f}  <- far weaker: CAPE is NOT a timer")

    print("\n  next-10y real return by CAPE bucket:")
    print(st.bucket_returns(d["cape"], d["fwd10"]).apply(lambda x: f"{x:+.1%}").to_string().replace("\n", "\n  "))

    b0, b1 = st.implied_return(d["cape"], d["fwd10"])
    cur = d["cape"].dropna().iloc[-1]
    print(f"\n  fit: fwd10 = {b0:+.3f} + {b1:.2f}/CAPE  →  at CAPE {cur:.0f}, implied 10y real return {b0 + b1/cur:+.1%}")

    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(d[['cape']].dropna())}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
