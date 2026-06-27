"""Study 536 — Anomaly-Decay-Post-Publication.

A research-method demo (sister to Studies 343-350): take a handful of classic
cross-sectional anomalies, split each name's sample at the anomaly's *publication
year*, and measure how the long-short return shrinks AFTER publication — the
McLean & Pontiff (2016) decay. By construction this is a methodology demonstration
(None x Mirage), not a tradable edge.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
