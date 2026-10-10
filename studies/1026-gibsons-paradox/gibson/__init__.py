"""Study 1026 — Gibson's Paradox.

Keynes (1930) gave the name *Gibson's paradox* to a regularity that embarrassed monetary theory:
over a century and more of British data, long-term interest rates moved with the **price level**,
not with the **rate of inflation** that Fisher's theory says should drive them. Barsky & Summers
(1988) explained it as a gold-standard phenomenon. The investor's modern version is shorter:
"bond yields follow prices, so the price level tells you where yields go".

This study asks whether that version survives under fiat money, on frozen real tapes (Moody's
AAA yields, US core and headline CPI, a German long rate and deflator), and whether anything in
it can time duration once two trending series are prevented from correlating by accident.

- :mod:`gibson.data` — the pinned tapes, the past-only price gap, and a two-world synthetic
  generator (Fisher vs Gibson) with a ``signal_strength`` knob.
- :mod:`gibson.strategy` — unit roots, cointegration, the first-difference horse race,
  out-of-sample forecasts with Clark-West, the duration-timing backtest, and the verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
