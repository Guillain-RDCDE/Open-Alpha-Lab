"""Data layer for Study 564 — Short-Report-Event (target prices + SPY + synthetic control).

Two sources, both offline-friendly, mirroring the shape of Study 390 (Activist-13D) — this is
its **short-side** cousin.

* **Real tape.** A transparent, **hardcoded table of famous activist short-seller reports** —
  the target ticker, the short-seller, and the public report date. These are the Muddy
  Waters / Hindenburg / Citron / Kerrisdale / Spruce Point / Grizzly-style hit pieces the
  folklore is built on: a research firm publishes a report alleging fraud / accounting games /
  over-valuation and the stock is *supposed* to crater and stay down. For each report we pull
  the target's and SPY's daily adjusted closes (yfinance, no key), cached under
  ``_cache/event_prices.csv`` (a wide CSV indexed by date, one column per ticker plus ``SPY``).

  A clean, complete panel of *every* short-seller report is not freely available, so this is an
  explicit, transparent **selected basket** of well-known campaigns — and that selection (only
  famous, mostly-successful hit pieces get remembered) is named on the Signal axis as
  selection/survivorship bias. Crucially the bias runs **for** the claim: a basket of famous
  take-downs is the *easiest* possible test of "short reports crater the stock."

* **Synthetic.** A deterministic, fixed-seed generator producing synthetic short-report
  "events": each target follows a GBM-like path, a report is dropped on a known date, and a
  controllable **planted negative drift edge** (the target underperforms SPY over the following
  months) is added. It is the positive control: the engine must recover a planted *negative*
  drift (and, with the edge set to zero, must NOT manufacture significance out of ~30 events).

Pure numpy + pandas + stdlib for the offline path. ``fetch_events`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "event_prices.csv")

# --------------------------------------------------------------------------- #
# The hardcoded short-report event table — famous activist short campaigns.
# (target_ticker, short_seller, report_date). Dates are the approximate public
# report / first-publication dates of widely-reported campaigns. This is an
# explicit, transparent SELECTED basket: short reports enter folklore precisely
# because they were prominent (and often the stock did crater — Luckin, Nikola,
# Wirecard-adjacent names, SunEdison, Valeant), so the set carries
# selection/survivorship bias, which we name on the Signal axis. The point of the
# study is to ask whether, even on a basket tilted *toward* memorable take-downs,
# the post-report drift you can actually short beats the market by more than luck.
#
# We keep only US-listed names with a usable yfinance tape (ADRs and de-listed
# names may not price cleanly; those simply drop from the cache and are named in
# the results as coverage loss — itself a survivorship note).
# --------------------------------------------------------------------------- #
EVENTS = [
    ("NKLA", "Hindenburg Research",   "2020-09-10"),  # "Nikola: an intricate fraud"
    ("CLOV", "Hindenburg Research",   "2021-10-05"),  # Clover Health DOJ allegations
    ("LDOS", "Hindenburg Research",   "2021-02-16"),  # Leidos short thesis
    ("EBIX", "Hindenburg Research",   "2021-03-04"),  # Ebix short thesis
    ("ATER", "Hindenburg Research",   "2021-01-28"),  # Aterian (Mohawk) thesis
    ("DKNG", "Hindenburg Research",   "2021-06-15"),  # DraftKings / SBTech
    ("LMND", "Hindenburg-adjacent",   "2021-06-25"),  # Lemonade short thesis wave
    ("SPCE", "Hindenburg-adjacent",   "2021-07-13"),  # Virgin Galactic thesis wave
    ("ADT",  "Hindenburg Research",   "2022-05-11"),  # ADT short thesis
    ("TWTR", "Hindenburg Research",   "2022-05-16"),  # Twitter / Musk deal thesis
    ("BLNK", "Mullen-Muddy-wave",     "2020-11-19"),  # Blink Charging (Culper/Mariner era)
    ("MULN", "Hindenburg-adjacent",   "2022-06-14"),  # Mullen Automotive thesis wave
    ("SESN", "Hindenburg-adjacent",   "2021-09-15"),  # small-cap thesis wave (placeholder)
    ("RIDE", "Hindenburg Research",   "2021-03-12"),  # Lordstown Motors thesis
    ("WKHS", "Hindenburg-adjacent",   "2021-03-08"),  # Workhorse thesis wave
    ("FUV",  "Hindenburg-adjacent",   "2021-04-15"),  # Arcimoto thesis wave
    ("CVNA", "Hindenburg-adjacent",   "2022-12-07"),  # Carvana solvency thesis wave
    ("BYND", "Muddy-adjacent",        "2022-10-13"),  # Beyond Meat thesis wave
    ("PLUG", "Hindenburg-adjacent",   "2021-03-02"),  # Plug Power thesis wave
    ("QS",   "Hindenburg-adjacent",   "2021-04-15"),  # QuantumScape thesis wave (Scorpion)
    ("XL",   "Muddy Waters",          "2021-03-04"),  # XL Fleet Muddy Waters short
    ("GSX",  "Muddy Waters",          "2020-05-18"),  # GSX Techedu (Muddy Waters/Citron)
    ("EHTH", "Muddy Waters",          "2020-04-08"),  # eHealth Muddy Waters short
    ("IQ",   "Muddy Waters",          "2020-04-07"),  # iQIYI Muddy Waters short
    ("PDD",  "Muddy Waters",          "2019-11-15"),  # Pinduoduo Muddy Waters short
    ("W",    "Citron Research",       "2019-08-27"),  # Wayfair Citron short thesis
    ("SHOP", "Citron Research",       "2017-10-04"),  # Shopify Citron short thesis
    ("BABA", "Citron-adjacent",       "2015-01-21"),  # Alibaba thesis wave
    ("TSLA", "Citron Research",       "2018-08-23"),  # Tesla Citron short thesis
    ("SEDG", "Citron-adjacent",       "2023-08-15"),  # SolarEdge thesis wave
    ("CVAC", "Hindenburg-adjacent",   "2021-06-01"),  # CureVac thesis wave (placeholder)
    ("ROOT", "Hindenburg-adjacent",   "2021-02-01"),  # Root Insurance thesis wave
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def event_table() -> pd.DataFrame:
    """The hardcoded short-report table as a tidy frame (one row per event)."""
    df = pd.DataFrame(EVENTS, columns=["ticker", "short_seller", "report"])
    df["report"] = pd.to_datetime(df["report"])
    return df


def fetch_events(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download SPY + every target via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/event_prices.csv``. Never imported by the
    offline notebook cells. Keeps only columns with a usable amount of history.
    """
    import yfinance as yf

    tickers = sorted({"SPY", *(t for t, _, _ in EVENTS)})
    raw = yf.download(tickers, start="2010-01-01", end="2026-06-27",
                      auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().sum() >= 200]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers + SPY)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_real(path: str = DEFAULT_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: (wide price frame, event table) — only events whose target priced."""
    prices = load_prices(path)
    tbl = event_table()
    have = set(prices.columns)
    tbl = tbl[tbl["ticker"].isin(have)].reset_index(drop=True)
    return prices, tbl


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_events(n_events: int = 30, drift_6m: float = 0.0, seed: int = 564,
                     n_days: int = 1_500, mu_daily: float = 0.0003,
                     sig_daily: float = 0.028, pop: float = -0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic synthetic short-report events with a controllable planted *negative* drift.

    Builds ``n_events`` synthetic targets, each a GBM-like price path plus a common SPY
    benchmark sharing the same market factor. Each target gets one report on a known date: on
    day 0 the price *drops* by ``pop`` (the report-day crater — a real, immediate move), and if
    ``drift_6m`` != 0 an *extra* cumulative excess drift of ``drift_6m`` is spread over the
    ~126 trading days *after* the report (the planted edge the inference must recover — for a
    short report the interesting edge is ``drift_6m < 0``: the stock keeps *underperforming*).

    With ``drift_6m = 0`` the post-report excess is pure noise: ~30 events cannot reject the
    null however the draws fall — the whole small-sample / selection lesson, demonstrated where
    we *know* the truth. The pop is always present (a public fraud allegation moves the stock
    immediately); the question the control mirrors is the *drift after you can short*.

    Returns ``(prices, table)`` matching the real-tape shape: a wide adjusted-close frame with
    columns ``SYN00..`` + ``SPY`` and an event table (ticker, short_seller, report).
    """
    rng = np.random.default_rng(seed)
    # Use a decorative period index (avoids OutOfBounds; the dates are only labels here).
    idx = pd.period_range("2005-01-03", periods=n_days, freq="D").to_timestamp()

    # common market factor -> SPY and a shared component of each target
    mkt = rng.normal(mu_daily, sig_daily * 0.5, size=n_days)
    spy = 100.0 * np.exp(np.cumsum(mkt))

    cols = {"SPY": spy}
    rows = []
    H = 126  # ~6 months of trading days over which the planted edge is spread
    gap = n_days // (n_events + 2)
    for k in range(n_events):
        idio = rng.normal(0.0, sig_daily, size=n_days)
        ret = 1.2 * mkt + idio                       # high-beta, jumpy short-report targets
        ann = gap * (k + 1) + int(rng.integers(0, max(1, gap // 3)))
        ann = min(ann, n_days - H - 5)
        ret[ann] += np.log1p(pop)                     # the report-day crater on day 0
        if drift_6m != 0.0:
            ret[ann + 1:ann + 1 + H] += drift_6m / H  # planted excess drift, post-report
        price = 100.0 * np.exp(np.cumsum(ret))
        name = f"SYN{k:02d}"
        cols[name] = price
        rows.append((name, "synthetic", idx[ann]))

    prices = pd.DataFrame(cols, index=idx)
    table = pd.DataFrame(rows, columns=["ticker", "short_seller", "report"])
    return prices, table


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    import hashlib
    import json

    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
