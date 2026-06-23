"""Data layer for Study 379 — ETF Lead-Lag (leader ETF + a basket of slower members).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for the leader ``SPY`` and a basket of smaller,
  less-liquid US members (mid/small-cap names that update more slowly than the headline
  index), via yfinance (no key), cached under ``_cache/lead_lag_prices.csv`` (a wide CSV
  indexed by date, one column per ticker plus ``SPY``). From these we build daily log
  returns: the leader return and the cross-sectional **laggard basket** return (equal-weight
  mean of the member returns each day).

* **Synthetic.** A deterministic, fixed-seed generator that builds a leader return series and
  a member series that follows the leader with a **planted one-day lead** controlled by a
  single knob ``lead`` (the fraction of yesterday's leader move that leaks into today's member
  return). It is the positive control: with ``lead=0`` the inference must NOT manufacture
  significance; with a large planted ``lead`` the next-bar rule must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_basket`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "lead_lag_prices.csv")

# Leader = the big, liquid headline ETF. Members = a fixed basket of smaller, slower,
# less-liquid US names (mid/small-cap, lower dollar-volume) — the believers' "slow members"
# that the leader is supposed to drag along a day later. A daily-bar basket is a PROXY for
# true intraday lead-lag (named on the Signal axis); every input is a public adjusted close.
LEADER = "SPY"
MEMBERS = [
    # mid/small-cap, lower-liquidity, long-listed names (slower to digest index-level news)
    "LEG", "SEE", "PBI", "MAT", "HRB", "NWL", "IPG", "RHI", "JNPR", "FFIV",
    "AKAM", "WHR", "HOG", "GT", "MOS", "CF", "FMC", "ALB", "VNO", "BXP",
    "HST", "KIM", "REG", "FRT", "UHS", "DVA", "XRAY", "ZBH", "PNR", "AOS",
    "SNA", "DOV", "ROL", "TXT", "BWA", "GPC", "AAP", "TPR", "RL", "PVH",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_basket(start: str = "2000-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the leader + member basket via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/lead_lag_prices.csv``. Never imported by the
    offline notebook cells. Keeps only columns with a long, mostly-complete history.
    """
    import yfinance as yf

    tickers = [LEADER] + MEMBERS
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    # keep names present for >=70% of the window (long-listed, avoids ragged starts)
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.70]
    if LEADER not in keep:
        keep = [LEADER] + keep
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = members + leader)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def lead_lag_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the daily lead-lag frame from prices.

    Returns a frame indexed by date with columns:
      * ``leader`` — leader (SPY) daily log return
      * ``members`` — equal-weight mean of member daily log returns (the laggard basket)

    Days where fewer than 10 member names have a return are dropped (early years before the
    basket fills in). All returns are log returns of adjusted closes.
    """
    cols = [c for c in prices.columns if c != LEADER]
    members = prices[cols]
    leader = prices[LEADER]
    lret = np.log(leader).diff()
    mret = np.log(members).diff()
    enough = mret.notna().sum(axis=1) >= 10
    basket = mret.mean(axis=1, skipna=True).where(enough)
    out = pd.DataFrame({"leader": lret, "members": basket})
    out = out.dropna()
    return out


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: cached prices -> lead-lag (leader/members return) frame in one call."""
    return lead_lag_frame(load_prices(path))


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_leadlag(n_days: int = 6_000, lead: float = 0.0, seed: int = 379,
                      sig_daily: float = 0.010, beta: float = 1.0,
                      idio: float = 0.008) -> pd.DataFrame:
    """Deterministic leader/member returns with a PLANTED one-day lead.

    The leader return is i.i.d. Gaussian. The member basket return on day ``t`` is::

        members[t] = beta * leader[t]            # contemporaneous co-movement (no edge)
                     + lead  * leader[t-1]        # the PLANTED one-day lead (the edge knob)
                     + idio * noise[t]            # idiosyncratic basket noise

    ``lead`` is the fraction of *yesterday's* leader move that leaks into *today's* member
    return — exactly the believers' "slow members catch up a day later." With ``lead = 0``
    there is no next-day predictability and the inference must NOT manufacture significance;
    with a large ``lead`` the next-bar rule and the cross-correlation profile must light up.

    The contemporaneous ``beta`` term is deliberately large: members move *with* the leader
    today (that is not tradable — you can't trade on information you don't yet have). Only the
    ``lead`` term is a genuine, exploitable next-day edge.
    """
    rng = np.random.default_rng(seed)
    leader = rng.normal(0.0003, sig_daily, size=n_days)
    noise = rng.normal(0.0, 1.0, size=n_days)
    members = np.empty(n_days)
    members[0] = beta * leader[0] + idio * noise[0]
    members[1:] = beta * leader[1:] + lead * leader[:-1] + idio * noise[1:]
    # period_range keeps a long daily series safe from OutOfBoundsDatetime; the index here is
    # only a decorative label (the engine works on the ordered return arrays).
    idx = pd.period_range("1990-01-01", periods=n_days, freq="D").to_timestamp()
    return pd.DataFrame({"leader": leader, "members": members}, index=idx)
