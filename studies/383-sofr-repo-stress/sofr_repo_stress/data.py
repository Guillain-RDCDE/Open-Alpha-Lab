"""Data layer for Study 383 — SOFR-Repo-Stress (funding spikes vs risk assets).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for the risk assets the folklore says the repo
  spike warns: ``SPY`` (US equities), ``HYG`` (high-yield credit) and ``LQD``
  (investment-grade credit), via yfinance (no key), cached under
  ``_cache/risk_assets.csv``. We pair this tape with a **hardcoded, sourced table of
  named repo-stress episodes** (:data:`REPO_EPISODES`) — there is no free, clean daily
  SOFR / repo-spread series on yfinance, so the *signal* itself is the curated list of
  well-documented funding-stress dates, and we measure the forward reaction of the risk
  assets around each.

* **Synthetic.** A deterministic, fixed-seed generator that produces a risk-asset price
  series and a set of "stress dates" *with a controllable planted forward edge*. It is
  the positive control: the inference must recover the planted edge (and, with the edge
  set to zero, must NOT manufacture significance out of a dozen events).

Pure numpy + pandas + stdlib for the offline path. ``fetch_risk_assets`` (network) is
only used once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "risk_assets.csv")

# The risk assets the repo-spike folklore claims to warn: equities + credit.
RISK_ASSETS = ["SPY", "HYG", "LQD"]

# --------------------------------------------------------------------------- #
# The hardcoded, sourced table of named repo / funding-stress episodes.
# --------------------------------------------------------------------------- #
# Each row: (date the stress *peaked / was first widely flagged*, a short tag, a one-line
# source note). Dates are the day the overnight repo / SOFR spike was most acute and most
# reported; we enter SPY/credit one trading day later (no look-ahead). This is the
# canonical "watch the repo market" list assembled from contemporaneous Fed / press
# coverage — a curated SIGNAL, not a fabricated price series. It is deliberately
# *generous*: it includes the quarter/year-end turn spikes the lore also points to, which
# makes the event count as large as honesty allows (and the test as powerful as possible).
REPO_EPISODES = [
    ("2018-12-31", "year-end-2018", "Year-end 2018 funding turn; GC repo spiked into the turn"),
    ("2019-03-29", "quarter-end-2019Q1", "Quarter-end repo pressure, Mar 2019"),
    ("2019-09-17", "sept-2019", "THE episode: SOFR ~5.25%, overnight GC repo printed ~10%; Fed restarted repo ops"),
    ("2019-12-31", "year-end-2019", "Year-end 2019 turn; Fed pre-emptive term repo ops"),
    ("2020-03-16", "covid-mar-2020", "March 2020 dash-for-cash; funding markets seized (COVID, a confound)"),
    ("2020-09-30", "quarter-end-2020Q3", "Quarter-end Sep 2020 funding pressure"),
    ("2021-12-31", "year-end-2021", "Year-end 2021 turn amid record reverse-repo usage"),
    ("2022-09-30", "quarter-end-2022Q3", "Sep 2022 quarter-end; rate-hike-cycle funding stress"),
    ("2023-03-10", "svb-mar-2023", "SVB / regional-bank stress; discount-window + BTFP funding surge"),
    ("2023-09-29", "quarter-end-2023Q3", "Sep 2023 quarter-end SOFR firmness, QT-drained reserves"),
    ("2024-09-30", "quarter-end-2024Q3", "Sep 2024 quarter-end SOFR spike, reserves scarcity debate"),
    ("2024-12-31", "year-end-2024", "Year-end 2024 turn; SOFR/repo firmed as reserves fell"),
    ("2025-03-31", "quarter-end-2025Q1", "Mar 2025 quarter-end SOFR pressure"),
]

# A tighter sub-list: only the headline, *systemic* funding-stress events the strongest
# version of the claim rests on (drop the routine quarter/year-end turns). Used for a
# robustness pass — fewer, "purer" events, an even weaker test.
SYSTEMIC_ONLY = ["sept-2019", "covid-mar-2020", "svb-mar-2023"]


def episode_dates() -> pd.DatetimeIndex:
    """The full hardcoded list of repo-stress dates (the curated signal)."""
    return pd.DatetimeIndex([pd.Timestamp(d) for d, _, _ in REPO_EPISODES])


def episode_table() -> pd.DataFrame:
    """The episode table as a frame (date, tag, note) for display."""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "tag": t, "note": n} for d, t, n in REPO_EPISODES]
    ).set_index("date")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_risk_assets(start: str = "2010-01-01", end: str | None = None,
                      path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download SPY + credit ETFs via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/risk_assets.csv``. Never imported by the
    offline notebook cells.
    """
    import yfinance as yf

    raw = yf.download(RISK_ASSETS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    out = raw[RISK_ASSETS].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = risk assets)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: the cached risk-asset price frame (SPY/HYG/LQD adjusted closes)."""
    df = load_prices(path)
    return df.dropna(how="all")


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_tape(n_days: int = 4_000, n_events: int = 13,
                   edge_20d: float = 0.0, seed: int = 383,
                   mu_daily: float = 0.0003, sig_daily: float = 0.011) -> dict:
    """Deterministic risk-asset tape + ``n_events`` planted 'repo-stress' dates.

    Builds a single SPY-like price series with daily drift ``mu_daily`` and vol
    ``sig_daily``, then scatters ``n_events`` non-overlapping "stress" dates. If
    ``edge_20d`` != 0 an *extra* cumulative drift of ``edge_20d`` (e.g. a negative number
    for a planted *bearish* reaction) is spread over the ~21 trading days **following**
    each stress date — the planted forward edge the inference must recover.

    With ``edge_20d = 0`` the control must NOT find significance: ~13 events cannot reject
    the null however the noise falls — that is the whole sample-size lesson, demonstrated
    on data where we *know* the truth. Returns ``{"prices": DataFrame, "events":
    DatetimeIndex}``.
    """
    rng = np.random.default_rng(seed)
    ret = rng.normal(mu_daily, sig_daily, size=n_days)

    # scatter non-overlapping stress dates, leaving room for the forward window
    span = n_days // (n_events + 2)
    starts = []
    for k in range(n_events):
        s = span * (k + 1) + int(rng.integers(0, span // 4))
        starts.append(s)
        if edge_20d != 0.0:
            h = 21                              # spread the planted edge over ~1 month
            lo, hi = s + 1, min(s + 1 + h, n_days)
            ret[lo:hi] += edge_20d / h

    price = 100.0 * np.exp(np.cumsum(ret))
    # a decorative business-day index well inside pandas' representable range
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    prices = pd.DataFrame({"SPY": price, "HYG": price, "LQD": price}, index=idx)
    events = pd.DatetimeIndex([idx[s] for s in starts])
    return {"prices": prices, "events": events}
