"""Data layer for Study 783 — IPO-Deal-Of-Year.

The claim under test: **the banks' celebrated "IPO of the year" then underperforms.**
Every year the league-table press (IFR / *International Financing Review*, and the
financial media generally) crowns a headline IPO — the biggest, splashiest, most
oversubscribed debut of the year. The folklore, an old cousin of Ritter's (1991) IPO
long-run underperformance, says the loudest deals are precisely the ones that then lag:
hype prices the pop, the sell-side victory-lap marks the top, and the newly-public stock
drifts down against the tape. We put a basket of these marquee US debuts on the stand and
measure their **forward abnormal return** (name minus SPY) over 3, 6 and 12 months from the
first trading close.

Four ingredients:

* **The marquee-IPO calendar, hardcoded.** A basket of the most celebrated, "deal of the
  year"-class US listings 2012->2024 — the debuts that dominated the financial press and
  the league tables in their year (Facebook, Alibaba, Snap, Uber, Beyond Meat, Airbnb,
  DoorDash, Coinbase, Rivian, Arm, Reddit, ...). Each row is a **real ticker with its real
  first-trade date**, cross-checked against exchange notices and contemporaneous coverage.

  **SELECTION CAVEAT (read this).** This basket is chosen *ex post* — these are names we
  remember *because* they were the year's marquee deal. That is a deliberate, honest bias:
  the test is descriptive ("did the celebrated debuts underperform?"), not a live tradable
  rule (you cannot buy "the IPO of the year" at its open — the crown is awarded later). The
  verdict reflects this: even a clean negative drift here is a *Mirage* to trade.

* **The benchmark (yfinance).** ``SPY`` (S&P 500 total return). We measure each name's
  **abnormal** forward return (name minus SPY), not its raw move, so a rising or falling
  broad tape is netted out.

* **No fundamental proxy needed.** "Post-IPO underperformance" is a pure price-path claim
  anchored on the first trading day — nothing is reconstructed from filings.

* **Synthetic world.** A deterministic, seeded paired (name, benchmark) log-return world
  with a TUNABLE planted post-IPO forward drift on a synthetic IPO calendar. ``bump = 0`` is
  the null world; the one-sample-t machinery must not manufacture significance from it.

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
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The marquee "IPO of the year"-class US-listed debut calendar, hardcoded: ticker, real
# first-trade date, and the deal's shorthand. These are the splashiest, league-table-topping
# listings of their year (the debuts the sell-side and financial press crowned). Dates are
# the first regular-way trading session, cross-checked against exchange notices and
# contemporaneous coverage. SELECTION IS EX POST BY DESIGN — see the module docstring.
# --------------------------------------------------------------------------- #
EVENTS = [
    # ticker, first_trade_date, deal shorthand
    ("META", "2012-05-18", "Facebook — largest tech IPO of 2012"),
    ("BABA", "2014-09-19", "Alibaba — largest IPO ever at the time (2014)"),
    ("SNAP", "2017-03-02", "Snap — marquee consumer-tech debut of 2017"),
    ("LYFT", "2019-03-29", "Lyft — first ride-hail listing, 2019"),
    ("BYND", "2019-05-02", "Beyond Meat — best-performing IPO pop of 2019"),
    ("UBER", "2019-05-10", "Uber — most-anticipated deal of 2019"),
    ("PLTR", "2020-09-30", "Palantir — headline direct listing, 2020"),
    ("DASH", "2020-12-09", "DoorDash — marquee pandemic debut, 2020"),
    ("ABNB", "2020-12-10", "Airbnb — deal of the year, 2020"),
    ("RBLX", "2021-03-10", "Roblox — headline direct listing, 2021"),
    ("COIN", "2021-04-14", "Coinbase — crypto's landmark listing, 2021"),
    ("HOOD", "2021-07-29", "Robinhood — retail-era marquee IPO, 2021"),
    ("RIVN", "2021-11-10", "Rivian — biggest US IPO of 2021"),
    ("ARM",  "2023-09-14", "Arm — largest IPO of 2023"),
    ("CART", "2023-09-19", "Instacart — headline consumer debut, 2023"),
    ("KVYO", "2023-09-20", "Klaviyo — marquee SaaS listing, 2023"),
    ("RDDT", "2024-03-21", "Reddit — most-watched IPO of 2024"),
]


def all_tickers() -> list[str]:
    return [t for t, _, _ in EVENTS] + [BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"ipoy_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2011-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for every marquee name + SPY; cache.

    Retries with linear backoff — Yahoo rate-limits transient bursts, so a first empty frame
    is usually cured by a short wait rather than a real "no such ticker".
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
# Synthetic world -- planted post-IPO forward drift
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 802, n_events: int = 17,
                    n_days: int = 6000, spacing: int = 300, horizon: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (name, benchmark) log-return world with a planted post-IPO
    forward drift.

    Both series are correlated (rho ~ 0.55, like a single high-beta new listing vs SPY)
    zero-mean noise; starting the session AFTER each synthetic "IPO day" (every
    ``spacing``-th business day) the name gets an EXTRA ``bump / horizon`` log-return on each
    of the next ``horizon`` sessions -- a planted forward drift whose cumulative size over
    the forward window is ~``bump``. ``bump = 0`` is the null world; ``bump < 0`` plants
    underperformance, ``bump > 0`` outperformance.

    Business-day integer index (positions 0..n_days). Returns
    (name_logret, bench_logret, ipo_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.55
    common = rng.normal(0.0, 0.011, n_days)
    idio_a = rng.normal(0.0, 0.018, n_days)   # new listings are volatile
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    ipo_pos = list(range(spacing, n_days - horizon - 5, spacing))[:n_events]
    drift = bump / horizon
    for p in ipo_pos:
        a[p + 1:p + 1 + horizon] += drift   # planted forward drift over the window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), ipo_pos
