"""Study 67 — Fed-Drift: the pre-FOMC announcement drift on the S&P 500.

Lucca & Moench (2015): equities drift up in the ~24 hours before a scheduled FOMC statement. The effect
was real and economically enormous (pre-2011, ~3% of sessions earned ~20% of the market's entire
cumulative return) — but it decayed sharply after publication (post-2011 the pre-FOMC day is barely
above a normal day, t ≈ 0.4). A real anomaly, arbitraged away in plain sight.
"""
from . import data, strategy  # noqa: F401
