"""Study 840 — Clustered Standard Errors (cross-sectional dependence).

Petersen (2009); Fama & MacBeth (1973); Moulton (1986): in a panel with a **common time
effect** — a shock that hits every firm in a period at once — the residuals of a pooled
regression are correlated **across firms within a period**. The ordinary i.i.d. OLS standard
error, and one-way **firm** clustering, ignore that dependence and are far too small, so the
naive *t*-statistic **overstates significance**. **Fama-MacBeth** and **time / two-way
clustering** restore calibration. We build a mean-zero null panel with a tunable common time
factor and show the naive false-positive rate blows past 5% (and firm-clustering does **not**
help) while Fama-MacBeth stays calibrated, quantifying the *t*-inflation against its closed
form — the Moulton factor ``sqrt(1 + (N-1)*rho_x*rho_e)``.

* ``data``     — deterministic seeded null-panel generator (a common time factor in both the
                 regressor and the residual, with a ``beta`` knob for the positive control) and
                 the closed-form Moulton inflation factor. No network, no market data.
* ``strategy`` — the four standard errors for one pooled slope (naive OLS, firm-clustered,
                 time-clustered, Fama-MacBeth), their vectorised Monte-Carlo forms, and the
                 calibration / false-positive / inflation-curve / control / power experiments.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
