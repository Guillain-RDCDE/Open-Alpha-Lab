"""Data layer for Study 284 (Equinox-Effect).

Two components, both fully offline for the deterministic core:

- ``EQUINOX_SOLSTICE`` — a hardcoded table of every March/September equinox and
  June/December solstice from 1928 (start of our ^GSPC daily cache) through 2026.
  Each row records the calendar date (UTC day of the astronomical instant), the
  ``kind`` ("equinox" or "solstice"), and the ``season`` it ushers in for the
  Northern Hemisphere ("spring"/"summer"/"autumn"/"winter"). These instants are
  computed once from Meeus' *Astronomical Algorithms* (ch. 27) — accurate to the
  minute against published tables — and hardcoded here to keep the core fully
  offline and reproducible. Source: Jean Meeus, *Astronomical Algorithms* (2nd
  ed., 1998); cross-checked against the US Naval Observatory's "Earth's Seasons"
  table and timeanddate.com.

- ``synthetic_panel`` — a deterministic, offline daily-return generator with an
  optional planted "equinox/solstice premium/drag" knob. ``signal_bps = 0`` is the
  null (no effect), so tests confirm the machinery is truthful before we look at
  real data.

- ``fetch_gspc_daily`` — real S&P 500 (^GSPC) daily price returns from the
  repo-level cache (``_cache/^GSPC_split_only.parquet``). Cache-only by default
  (``fetch=False``); ``fetch=True`` triggers a lazy yfinance download. No network
  is touched in the default path.

No look-ahead: an event window is built around each equinox/solstice *date*, and
the "event-day return" is the close-to-close return realised on the first trading
day on or after the astronomical instant — the soonest a trader could have acted.
The event study is symmetric (``win`` trading days either side) and uses only
realised returns, never future information beyond the window the reader is shown.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

SEED = 284

# ---------------------------------------------------------------------------
# The hardcoded equinox/solstice table — the study's deterministic, offline core
# ---------------------------------------------------------------------------
# Columns:
#   date   : ISO calendar date (UTC day) of the astronomical instant
#   kind   : "equinox" (March / September) or "solstice" (June / December)
#   season : Northern-Hemisphere season ushered in
#            ("spring" Mar, "summer" Jun, "autumn" Sep, "winter" Dec)
#
# Computed from Meeus, *Astronomical Algorithms* (2nd ed.), chapter 27, using the
# mean-instant polynomial plus the 24-term periodic correction (S) and the
# delta-lambda solar-anomaly factor. Verified to the minute against published
# USNO / timeanddate.com seasons tables (e.g. 2024: Mar 20, Jun 20, Sep 22,
# Dec 21 — all exact). As-of: 2026-06-17. Restricted to 1928–2026, the span of
# our ^GSPC daily cache.

EQUINOX_SOLSTICE: list[dict] = [
    {"date": "1928-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1928-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1928-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1928-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1929-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1929-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1929-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1929-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1930-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1930-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1930-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1930-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1931-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1931-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1931-09-24", "kind": "equinox", "season": "autumn"},
    {"date": "1931-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1932-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1932-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1932-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1932-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1933-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1933-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1933-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1933-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1934-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1934-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1934-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1934-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1935-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1935-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1935-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1935-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1936-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1936-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1936-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1936-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1937-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1937-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1937-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1937-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1938-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1938-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1938-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1938-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1939-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1939-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1939-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1939-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1940-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1940-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1940-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1940-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1941-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1941-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1941-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1941-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1942-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1942-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1942-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1942-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1943-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1943-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1943-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1943-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1944-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1944-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1944-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1944-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1945-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1945-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1945-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1945-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1946-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1946-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1946-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1946-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1947-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1947-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1947-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1947-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1948-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1948-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1948-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1948-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1949-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1949-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1949-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1949-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1950-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1950-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1950-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1950-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1951-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1951-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1951-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1951-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1952-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1952-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1952-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1952-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1953-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1953-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1953-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1953-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1954-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1954-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1954-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1954-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1955-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1955-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1955-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1955-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1956-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1956-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1956-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1956-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1957-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1957-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1957-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1957-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1958-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1958-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1958-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1958-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1959-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1959-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1959-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1959-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1960-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1960-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1960-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1960-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1961-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1961-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1961-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1961-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1962-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1962-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1962-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1962-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1963-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1963-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1963-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1963-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1964-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1964-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1964-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1964-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1965-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1965-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1965-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1965-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1966-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1966-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1966-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1966-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1967-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1967-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1967-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1967-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1968-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1968-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1968-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1968-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1969-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1969-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1969-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1969-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1970-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1970-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1970-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1970-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1971-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1971-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1971-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1971-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1972-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1972-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1972-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1972-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1973-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1973-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1973-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1973-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1974-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1974-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1974-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1974-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1975-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1975-06-22", "kind": "solstice", "season": "summer"},
    {"date": "1975-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1975-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1976-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1976-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1976-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1976-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1977-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1977-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1977-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1977-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1978-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1978-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1978-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1978-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1979-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1979-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1979-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1979-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1980-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1980-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1980-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1980-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1981-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1981-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1981-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1981-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1982-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1982-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1982-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1982-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1983-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1983-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1983-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1983-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1984-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1984-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1984-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1984-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1985-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1985-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1985-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1985-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1986-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1986-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1986-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1986-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1987-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1987-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1987-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1987-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1988-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1988-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1988-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1988-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1989-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1989-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1989-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1989-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1990-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1990-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1990-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1990-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1991-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1991-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1991-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1991-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1992-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1992-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1992-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1992-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1993-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1993-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1993-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1993-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1994-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1994-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1994-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1994-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1995-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1995-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1995-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1995-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1996-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1996-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1996-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1996-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1997-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1997-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1997-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "1997-12-21", "kind": "solstice", "season": "winter"},
    {"date": "1998-03-20", "kind": "equinox", "season": "spring"},
    {"date": "1998-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1998-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1998-12-22", "kind": "solstice", "season": "winter"},
    {"date": "1999-03-21", "kind": "equinox", "season": "spring"},
    {"date": "1999-06-21", "kind": "solstice", "season": "summer"},
    {"date": "1999-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "1999-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2000-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2000-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2000-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2000-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2001-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2001-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2001-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2001-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2002-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2002-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2002-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2002-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2003-03-21", "kind": "equinox", "season": "spring"},
    {"date": "2003-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2003-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2003-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2004-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2004-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2004-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2004-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2005-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2005-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2005-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2005-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2006-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2006-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2006-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2006-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2007-03-21", "kind": "equinox", "season": "spring"},
    {"date": "2007-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2007-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2007-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2008-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2008-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2008-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2008-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2009-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2009-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2009-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2009-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2010-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2010-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2010-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2010-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2011-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2011-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2011-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2011-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2012-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2012-06-20", "kind": "solstice", "season": "summer"},
    {"date": "2012-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2012-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2013-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2013-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2013-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2013-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2014-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2014-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2014-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2014-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2015-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2015-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2015-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2015-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2016-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2016-06-20", "kind": "solstice", "season": "summer"},
    {"date": "2016-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2016-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2017-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2017-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2017-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2017-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2018-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2018-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2018-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2018-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2019-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2019-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2019-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2019-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2020-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2020-06-20", "kind": "solstice", "season": "summer"},
    {"date": "2020-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2020-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2021-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2021-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2021-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2021-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2022-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2022-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2022-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2022-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2023-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2023-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2023-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2023-12-22", "kind": "solstice", "season": "winter"},
    {"date": "2024-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2024-06-20", "kind": "solstice", "season": "summer"},
    {"date": "2024-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2024-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2025-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2025-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2025-09-22", "kind": "equinox", "season": "autumn"},
    {"date": "2025-12-21", "kind": "solstice", "season": "winter"},
    {"date": "2026-03-20", "kind": "equinox", "season": "spring"},
    {"date": "2026-06-21", "kind": "solstice", "season": "summer"},
    {"date": "2026-09-23", "kind": "equinox", "season": "autumn"},
    {"date": "2026-12-21", "kind": "solstice", "season": "winter"},
]

EQUINOX_SOLSTICE_DF: pd.DataFrame = pd.DataFrame(EQUINOX_SOLSTICE)
EQUINOX_SOLSTICE_DF["date"] = pd.to_datetime(EQUINOX_SOLSTICE_DF["date"])


def equinox_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded equinox/solstice table.

    Columns: ``date`` (datetime64), ``kind`` ('equinox'|'solstice'),
    ``season`` ('spring'|'summer'|'autumn'|'winter').
    """
    return EQUINOX_SOLSTICE_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic daily-return generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    start: str = "1928-01-01",
    end: str = "2026-09-30",
    signal_bps: float = 0.0,
    base_bps: float = 3.0,
    vol_bps: float = 100.0,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with an optional planted equinox effect.

    Each *business* day is an i.i.d. draw N(base_bps, vol_bps^2) in basis points.
    On the first trading day on/after each equinox/solstice in
    ``EQUINOX_SOLSTICE``, an extra drift of ``signal_bps`` bps is added.
    ``signal_bps = 0`` is the null — equinox/solstice days are statistically
    identical to every other day. This is the study's null in a bottle.

    Returns ``(df, truth)`` where ``df`` is indexed by date with a single ``ret``
    column (simple daily return), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, end=end)
    rets = (base_bps + rng.normal(0.0, vol_bps, len(idx))) * 1e-4
    df = pd.DataFrame({"ret": rets}, index=idx)
    df.index.name = "Date"

    if signal_bps != 0.0:
        ev = equinox_table()
        for d in ev["date"]:
            pos = df.index.searchsorted(d)  # first trading day on/after the instant
            if 0 <= pos < len(df):
                df.iloc[pos, df.columns.get_loc("ret")] += signal_bps * 1e-4

    truth = {
        "start": start,
        "end": end,
        "signal_bps": signal_bps,
        "base_bps": base_bps,
        "vol_bps": vol_bps,
        "seed": seed,
        "n_days": len(df),
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — ^GSPC daily returns (repo-level cache, read-only by default)
# ---------------------------------------------------------------------------
def fetch_gspc_daily(
    cache_dir: str = REPO_CACHE,
    fetch: bool = False,
    start: str = "1928-01-01",
    end: str = "2026-09-30",
) -> pd.DataFrame:
    """Load ^GSPC daily close-to-close returns.

    Cache-only by default: reads ``_cache/^GSPC_split_only.parquet`` (staged in
    the repo-level cache, read-only). With ``fetch=True`` it lazily imports
    yfinance and downloads ^GSPC — the only path that touches the network.

    Returns a DataFrame indexed by ``Date`` (DatetimeIndex) with columns
    ``Close`` and ``ret`` (simple daily return).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    path = os.path.join(cache_dir, "^GSPC_split_only.parquet")
    if os.path.exists(path):
        px = pd.read_parquet(path)
    elif fetch:
        import yfinance as yf  # lazy import — network only on explicit request

        raw = yf.download("^GSPC", start=start, end=end, auto_adjust=False,
                          progress=False)
        px = raw[["Close"]].copy()
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
    else:
        raise FileNotFoundError(
            f"^GSPC cache not found at {path}. Pass fetch=True to download "
            "from yfinance, or stage the repo-level _cache/ parquet."
        )

    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)
    px = px[["Close"]].copy()
    px["ret"] = px["Close"].pct_change()
    px = px.loc[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
    return px.dropna()


def fingerprint(df: pd.DataFrame, col: str = "ret") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
