"""Study 863 — Treasury Noise Liquidity.

Hu, Pan & Wang (2013), *"Noise as Information for Illiquidity"* (Journal of Finance):
when arbitrage capital is abundant, smart money smooths the Treasury yield curve and
individual maturities hug a common smooth fit; when arbitrage capital is **scarce**
(illiquidity / funding stress), the curve gets **rough** — individual maturities drift
away from any smooth curve. Their **noise measure** is the root-mean-square deviation of
observed yields from a fitted smooth curve; high noise flags market-wide illiquidity and
has since become a canonical funding-stress gauge.

We build a *self-contained daily* version from four constant-maturity Treasury (CMT)
yields (13-week / 5-year / 10-year / 30-year) — the daily roughness = RMS deviation of
the four yields from a **quadratic-in-maturity** fit — and test whether **high-noise
regimes precede lower forward equity (SPY) and wider credit (HYG − IEF)**.

* ``data``     — the real tape (yfinance daily closes for ``^IRX/^FVX/^TNX/^TYX`` plus
                 ``SPY/HYG/IEF/LQD/TLT``, cached under this study's own ``_cache/`` as one
                 parquet), plus a deterministic seeded synthetic positive control with a
                 tunable planted noise→forward-return relation (null at ``edge=0``).
* ``strategy`` — the roughness (noise) construction, the predictive regressions of
                 forward SPY / HYG−IEF returns on noise, the inference primitives
                 (one-sample / Welch / Newey-West HAC / Wilson / OLS+HAC slope), a
                 block-bootstrap placebo, an era cut, a costed regime timer, and the
                 seeded synthetic detector.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
