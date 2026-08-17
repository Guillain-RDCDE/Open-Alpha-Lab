"""Shared fixtures — deterministic synthetic tapes for Study 931 (the CEF IPO hole).

Everything here is offline and deterministic (fixed seed 931; no network, no cache):

- ``planted_panel`` / ``null_panel`` — panels of staggered synthetic IPOs with and
  without a planted first-year slide.
- ``planted_frame`` / ``null_frame`` — the same panels reshaped into the *real-tape*
  shape the event study consumes: one wide price frame, an IPO list of
  ``(ticker, offer, sleeve, benchmark)`` tuples, and a raw-close frame. This lets the
  tests exercise :func:`cef_ipo.strategy.build_table` and everything downstream of it
  without ever touching ``studies/_cache``.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cef_ipo import data  # noqa: E402

OFFER = 20.0


def _as_frame(panel):
    """Reshape a synthetic panel into (prices, ipos, raw) in the real-tape shape."""
    cols = {}
    ipos = []
    for key, px in panel.items():
        cols[key] = px["fund"]
        cols[key + "_B"] = px["bench"]
        ipos.append((key, OFFER, "synthetic", key + "_B"))
    prices = pd.DataFrame(cols).sort_index()
    prices.index.name = "date"
    # Raw (unadjusted) closes: the synthetic fund IS its own raw close, so the day-one
    # gap from the offering price is whatever the generator produced on day one.
    raw = prices[[k for k, _, _, _ in ipos]].copy()
    return prices, tuple(ipos), raw


@pytest.fixture
def planted_panel():
    """A panel of staggered IPOs each carrying a planted −10% first-year slide."""
    return data.synthetic_panel(signal_strength=1.0, seed=931)


@pytest.fixture
def null_panel():
    """The null: same funds, same benchmarks, nothing planted."""
    return data.synthetic_panel(signal_strength=0.0, seed=931)


@pytest.fixture
def planted_frame(planted_panel):
    return _as_frame(planted_panel[0])


@pytest.fixture
def null_frame(null_panel):
    return _as_frame(null_panel[0])
