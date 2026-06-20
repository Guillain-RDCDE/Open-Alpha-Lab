"""Study 333 — Recovery-Speed (drawdown-recovery momentum).

After a deep, market-wide drawdown, do the names that *climb back fastest* keep
leading afterwards? The package exposes:

- ``data``     — a deterministic synthetic panel generator (plants a positive control
                 AND a null) plus a cache-first real-panel loader.
- ``strategy`` — the recovery-speed ranking engine, a long-short decile book, honest
                 stats (HAC/Newey-West t, circular block-bootstrap CI), costs, one lag.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
