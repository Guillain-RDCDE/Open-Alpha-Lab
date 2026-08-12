"""Study 867 — Currency Crash Risk (Brunnermeier-Nagel-Pedersen).

Brunnermeier, Nagel & Pedersen (2008), *"Carry Trades and Currency Crashes"*:
**high-carry currencies are exposed to crash risk** — their returns are **negatively
skewed** ("up by the stairs, down by the elevator"), the higher the carry the deeper the
skew, and a long-high / short-low carry basket inherits that crash tail. The carry
premium is compensation for a sold-crash exposure, not a free lunch.

* ``data``     — the real tape (yfinance daily FX spot for a fixed 8-currency basket vs
                 USD incl. the notorious high-carry MXN, cached under the study's own
                 ``_cache/`` as a weekly parquet; every pair normalised to
                 **USD-per-foreign-currency**) plus a deterministic seeded synthetic
                 positive control (a planted carry-crash skew, null at ``edge=0``).
* ``strategy`` — the carry basket, the realized-skew crash test (Newey-West *t* on the
                 standardised-cubed-residual series), the skew-carry cross-section, the
                 crash-conditional split, the label-shuffle placebo, the era cut, the
                 costed carry book, and the inference primitives.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
