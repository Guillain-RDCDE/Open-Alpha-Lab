"""Study 879 — Weekly Economic Index (Lewis-Mertens-Stock).

The **Weekly Economic Index (WEI)** blends **10 weekly** activity series (Redbook
same-store sales, initial & continuing unemployment-insurance claims, adjusted income-tax
withholding, railroad traffic, retail fuel sales, temporary-staffing, steel production,
electricity output, and consumer-confidence) into a single **real-time nowcast** of U.S.
year-over-year real activity, published every week by the **Dallas Fed** (originally NY
Fed; Lewis, Mertens & Stock, 2020). The claim under test: because the WEI carries
*higher-frequency* growth information than the monthly macro tape, its **level** and its
**weekly change** should predict **forward SPY** returns and the **cyclical-vs-defensive
rotation** (consumer-discretionary ``XLY`` vs consumer-staples ``XLP``).

* ``data``     — the real tape (Dallas Fed WEI workbook + SPY / XLY / XLP daily closes via
                 yfinance, cached under this study's own ``_cache/``) plus a deterministic,
                 seeded synthetic positive control (a planted WEI->forward-return relation,
                 null at ``edge = 0``).
* ``strategy`` — the weekly predictive regression of forward SPY / rotation returns on the
                 (level & weekly change of the) nowcast, the inference primitives
                 (one-sample / Welch / Newey-West HAC / Wilson / placebo), an era cut, and
                 the costed rotation overlay.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
