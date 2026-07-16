"""Data layer for Study 784 — Analyst-Cluster.

The claim under test: **fade a stock after a *cluster* of same-week analyst upgrades.**
The behavioural story is that a burst of "everyone upgrades at once" price-target hikes
marks a local sentiment top — the sell-side pile-in is the last marginal buyer — so the
name should *underperform* in the following weeks (a tradable short). The bullish
counter-story is the opposite: an upgrade wave is a real information event and the stock
*keeps drifting up* (sell-side momentum / post-earnings drift). This study lets the tape
decide, on a single high-profile name where upgrade clusters are frequent and loud.

* * *

## LABELLED PROXY — read this before trusting the "cluster" dates

There is **no free, survivorship-clean, point-in-time feed of individual analyst upgrades**
in this repo's offline toolbox, so we do **not** claim a live analyst tape. Instead the
event list below is a **LABELLED PROXY**:

    the cluster-of-upgrades week is proxied by NVDA's quarterly EARNINGS week.

Why that is an honest proxy and not a fabrication:

* NVDA's quarterly report is *the* canonical trigger of a same-week analyst pile-in — after
  every print, financial media reliably runs "N analysts raised their price targets on
  NVIDIA" (routinely 20-40 firms in the AI-boom years). The earnings week and the
  upgrade-cluster week are, in practice, the same week.
* The **earnings dates themselves are real, publicly-verifiable facts** (NVDA 8-K filings /
  earnings-history archives; see ``EVENTS`` below), and they are *known in advance* (NVDA
  pre-announces the report date ~3-4 weeks out), so a "short the K sessions after the
  cluster" rule is calendar-known and zero-look-ahead by construction.
* The tradeoff we are honest about: this proxy inherits **post-earnings-announcement drift**
  as a confound — a "fade after the cluster" and a "post-earnings drift" are measured on the
  same window. We report that openly in ``docs/results.md`` rather than hide it.

If you have a real point-in-time upgrade feed, swap ``EVENTS`` for the actual cluster dates;
the rest of the machinery is unchanged.

## The other ingredients

* **The tradable instrument (yfinance).** ``NVDA`` — NVIDIA Corp — benchmarked against
  ``SPY`` (S&P 500 total return) so the test measures NVDA's *abnormal* return, not the
  market's. NVDA is a very high-beta name, so a raw NVDA move over any window is largely
  just beta; ``NVDA − SPY`` is the abnormal leg throughout.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "pre-cluster run-up" and an optional "post-cluster fade" on a
  synthetic quarterly calendar. ``bump = fade = 0`` is the null world; the one-sample-t
  machinery must not manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete month at publication
INSTRUMENT = "NVDA"      # NVIDIA Corp
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# LABELLED PROXY event calendar. Each entry is (fiscal-quarter tag, date) where the DATE
# is NVDA's REAL quarterly earnings-release date — a real, publicly-verifiable fact (NVDA
# 8-K filings / earnings-history archives) — used as a PROXY for that quarter's same-week
# analyst-upgrade cluster (the post-print price-target-hike wave). Dates are known ~3-4
# weeks ahead, so "short the K sessions after" is calendar-known and zero-look-ahead.
# 39 clusters, 2016Q4-FY .. 2025Q2-FY, spanning the pre-boom and AI-boom regimes.
# --------------------------------------------------------------------------- #
EVENTS = [
    # tag,        earnings-week date (PROXY anchor for the upgrade cluster)
    ("2016-Feb", "2016-02-17"),
    ("2016-May", "2016-05-12"),
    ("2016-Aug", "2016-08-11"),
    ("2016-Nov", "2016-11-10"),
    ("2017-Feb", "2017-02-09"),
    ("2017-May", "2017-05-09"),
    ("2017-Aug", "2017-08-10"),
    ("2017-Nov", "2017-11-09"),
    ("2018-Feb", "2018-02-08"),
    ("2018-May", "2018-05-10"),
    ("2018-Aug", "2018-08-16"),
    ("2018-Nov", "2018-11-15"),
    ("2019-Feb", "2019-02-14"),
    ("2019-May", "2019-05-16"),
    ("2019-Aug", "2019-08-15"),
    ("2019-Nov", "2019-11-14"),
    ("2020-Feb", "2020-02-13"),
    ("2020-May", "2020-05-21"),
    ("2020-Aug", "2020-08-19"),
    ("2020-Nov", "2020-11-18"),
    ("2021-Feb", "2021-02-24"),
    ("2021-May", "2021-05-26"),
    ("2021-Aug", "2021-08-18"),
    ("2021-Nov", "2021-11-17"),
    ("2022-Feb", "2022-02-16"),
    ("2022-May", "2022-05-25"),
    ("2022-Aug", "2022-08-24"),
    ("2022-Nov", "2022-11-16"),
    ("2023-Feb", "2023-02-22"),
    ("2023-May", "2023-05-24"),
    ("2023-Aug", "2023-08-23"),
    ("2023-Nov", "2023-11-21"),
    ("2024-Feb", "2024-02-21"),
    ("2024-May", "2024-05-22"),
    ("2024-Aug", "2024-08-28"),
    ("2024-Nov", "2024-11-20"),
    ("2025-Feb", "2025-02-26"),
    ("2025-May", "2025-05-28"),
    ("2025-Aug", "2025-08-27"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"cluster_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2015-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for NVDA + SPY; cache them.

    Retries with linear backoff — Yahoo rate-limits transient bursts, so a first empty
    frame is usually cured by a short wait rather than a real "no such ticker".
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        last_err = None
        for attempt in range(retries):
            try:
                d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Close"]].dropna()
                if len(d) > 0:
                    d.to_csv(_cache_path(t))
                    break
                last_err = f"empty frame for {t}"
            except Exception as e:  # noqa: BLE001 -- transient network/rate-limit
                last_err = str(e)
            time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"fetch failed for {t} after {retries} tries: {last_err}")


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted pre-cluster run-up + optional post-cluster fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 803,
                    n_events: int = 39, n_days: int = 5000, spacing: int = 63,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    and an optional post-cluster fade.

    Both series are correlated (rho ~ 0.6, like a single high-beta name vs SPY) zero-mean
    noise; on the trading day just before each synthetic "cluster day" (every
    ``spacing``-th business day — a quarterly cadence) the asset gets an EXTRA ``bump``
    log-return -- a planted pre-cluster run-up -- and on the day just after, an EXTRA
    ``-fade`` -- a planted post-cluster fade (the tradable-short direction the folklore
    predicts). ``bump = fade = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, cluster_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.6
    common = rng.normal(0.0, 0.014, n_days)
    idio_a = rng.normal(0.0, 0.020, n_days)   # NVDA is noisier than AAPL was
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    key_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in key_pos:
        a[p - 1] += bump      # planted run-up: shows up in the pre-cluster window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted fade: shows up in the post-cluster window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
