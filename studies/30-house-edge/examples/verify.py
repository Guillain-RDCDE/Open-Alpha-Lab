"""Reproduce the real-data headline run (docs/results.md) — S&P 500, 1990–today.

    python examples/verify.py            # cache-only (offline); prints if cache present
    python examples/verify.py --fetch    # download ^GSPC + ^IRX from Yahoo, then run

Fetches the real index + short rate, runs the SAME vol-targeted contrarian book under idealized
vs honest (full-notional) financing, and prints the table that docs/results.md fingerprints.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows cp1252 consoles
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from house_edge import costs, data, extension, strategy


def main(fetch: bool) -> None:
    price, rate = data.fetch_index(fetch=fetch)
    if price.empty:
        print("No cached real data. Re-run with --fetch (needs network) to download ^GSPC + ^IRX.")
        return
    price = price.loc["1990":]
    rate = rate.reindex(price.index).ffill().fillna(0.03)
    e = strategy.exposure(price)
    rows = {
        "buy & hold (total return)": strategy.summary(costs.buy_and_hold(price)),
        "strat — idealized costs": strategy.summary(costs.net_returns(e, price, rate, mode="idealized")),
        "strat — HONEST costs": strategy.summary(costs.net_returns(e, price, rate, mode="honest")),
    }
    print(f"\nReal S&P 500, {price.index[0].date()} - {price.index[-1].date()} ({len(price)} days)\n")
    print(f"{'':28} {'CAGR':>6} {'Sharpe':>7} {'maxDD':>7} {'Calmar':>7}")
    for nm, s in rows.items():
        print(f"{nm:28} {s['cagr']*100:5.1f}% {s['sharpe']:7.2f} {s['max_drawdown']*100:6.1f}% {s['calmar']:7.2f}")
    he = costs.house_edge(e, price, rate)
    print(f"\nHouse edge (idealized − honest CAGR): {he['house_edge_ann']*100:.2f} pts/yr | avg exposure {he['avg_exposure']:.2f}")
    print("\nFinancing-markup sweep (honest):")
    print(extension.financing_sweep(price, rate).round(4).to_string())
    print("\nDrawdown protection:", {k: round(v, 3) for k, v in extension.drawdown_protection(price, rate).items()})

    try:
        import pandas as pd

        from quantlab import repro
        fp = repro.fingerprint(pd.DataFrame({"price": price, "rate": rate}))
        print(f"\nas-of {price.index[-1].date()} · input fingerprint {fp}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
