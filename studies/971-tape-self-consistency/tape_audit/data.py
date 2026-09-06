"""Data layer for Study 971 — Does the Tape Agree With Itself?.

This study needs the *same asset asked four different ways*, so it keeps four caches
per ticker under its own prefixes: daily total-return bars, daily **unadjusted** bars with the
dividend and split events attached, weekly bars and monthly bars — all from the same provider,
in the same call pattern, minutes apart.

Four caches per ticker, each written by ``fetch`` and read offline by its own loader:

- ``daily_tr``  — ``auto_adjust=True`` daily bars (split- and dividend-adjusted close);
- ``daily_raw`` — ``auto_adjust=False`` daily bars plus the ``Adj Close`` column and the
  dividend and split event columns, i.e. everything needed to *rebuild* the adjusted series;
- ``weekly``    — ``interval='1wk'``, adjusted;
- ``monthly``   — ``interval='1mo'``, adjusted.

They are fetched in one pass, minutes apart at most, so any disagreement between them is the
provider's, not a timing artefact — that is the whole design of the study, and it is the reason
the loaders refuse to mix a cache written on different days (``cache_dates``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Eight tapes with different corporate-action histories: splits (AAPL, NVDA, TSLA),
# large dividends (VYM, XLU), a fund that has done neither much (SPY, QQQ), and gold
# (no dividends at all, so it is the control).
TICKERS = ("SPY", "QQQ", "AAPL", "NVDA", "TSLA", "VYM", "XLU", "GLD")

AS_OF = "2026-06-30"
START = "1993-01-01"

FLAVOURS = ("daily_tr", "daily_raw", "weekly", "monthly")
_INTERVAL = {"daily_tr": "1d", "daily_raw": "1d", "weekly": "1wk", "monthly": "1mo"}
_ADJUST = {"daily_tr": True, "daily_raw": False, "weekly": True, "monthly": True}


def _cache_path(ticker: str, flavour: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"tapeaudit_{safe}_{flavour}.parquet")


def _meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "tapeaudit_fetch_meta.json")


def fetch(tickers=TICKERS, start: str = START, end: str | None = None,
          cache_dir: str = DEFAULT_CACHE, retries: int = 4) -> dict:
    """Download all four flavours for every ticker in one pass and cache them.

    Network-only. The fetch timestamp for each (ticker, flavour) is written alongside, so a
    later run can prove the four views were pulled from the same vintage of the provider's
    database rather than weeks apart.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    os.makedirs(cache_dir, exist_ok=True)
    meta = {}
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for tk in tickers:
        out[tk] = {}
        for fl in FLAVOURS:
            raw = None
            for _ in range(retries):
                try:
                    raw = yf.download(tk, start=start, end=end, interval=_INTERVAL[fl],
                                      auto_adjust=_ADJUST[fl], actions=(fl == "daily_raw"),
                                      progress=False)
                    if raw is not None and len(raw) > 0:
                        break
                except Exception:
                    time.sleep(2.0)
            if raw is None or len(raw) == 0:
                raise RuntimeError(f"yfinance returned no data for {tk} ({fl})")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "date"
            raw = raw[raw["close"].notna()]
            raw.to_parquet(_cache_path(tk, fl, cache_dir))
            meta[f"{tk}|{fl}"] = pd.Timestamp.utcnow().isoformat()
            out[tk][fl] = raw
    with open(_meta_path(cache_dir), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff all four flavours are cached for every ticker."""
    return all(os.path.exists(_cache_path(tk, fl, cache_dir))
               for tk in tickers for fl in FLAVOURS)


def load(ticker: str, flavour: str = "daily_tr", cache_dir: str = DEFAULT_CACHE,
         asof: str = AS_OF) -> pd.DataFrame:
    """Read one cached flavour offline, sliced to ``asof``."""
    path = _cache_path(ticker, flavour, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached {flavour} data for {ticker} at {path}. "
            f"Call tape_audit.data.fetch() once to populate the cache.")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)].sort_index()


def load_all(ticker: str, cache_dir: str = DEFAULT_CACHE,
             asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """``{flavour: frame}`` for one ticker."""
    return {fl: load(ticker, fl, cache_dir, asof) for fl in FLAVOURS}


def cache_dates(cache_dir: str = DEFAULT_CACHE) -> dict:
    """When each (ticker, flavour) was last fetched — the provenance of a disagreement."""
    p = _meta_path(cache_dir)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(df.select_dtypes("number").to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]




# --------------------------------------------------------------------------- #
# Synthetic tape — a known-good tape and a deliberately corrupted one
# --------------------------------------------------------------------------- #
def synthetic_tape(
    n_years: int = 10,
    vol_ann: float = 0.22,
    drift_ann: float = 0.08,
    dividend_yield: float = 0.02,
    n_dividends_per_year: int = 4,
    split_at: float | None = 0.6,     # fraction of the sample where a 2:1 split happens
    start: str = "2010-01-04",
    seed: int = 971,
) -> tuple[dict, dict]:
    """A KNOWN-GOOD tape: raw prices, dividends, a split, and the total-return series they imply.

    Returns ``(frames, truth)`` where ``frames`` has the same shape as the real cache —
    ``daily_tr`` (adjusted close), ``daily_raw`` (as-traded close plus ``dividends`` and
    ``stock_splits`` columns), ``weekly`` and ``monthly`` — all derived from one underlying
    path, so **every** consistency check in ``strategy.py`` must pass on it exactly. That is
    what makes the corrupted version below a meaningful test.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n)
    sd = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mu = drift_ann / TRADING_DAYS_PER_YEAR

    r_price = rng.normal(mu - dividend_yield / TRADING_DAYS_PER_YEAR, sd, n)
    price = 100.0 * np.exp(np.cumsum(r_price))

    div = np.zeros(n)
    step = max(1, n // (n_years * n_dividends_per_year))
    for i in range(step, n, step):
        div[i] = price[i] * dividend_yield / n_dividends_per_year

    # The provider's convention, reproduced deliberately: with ``auto_adjust=False`` the OHLC
    # is **already split-adjusted** and only dividends separate ``close`` from ``adj_close``.
    # (Pinned empirically in quantlab/data.py.) So a split leaves no jump in either price
    # series here — the split column records the event, and any tape where the price *does*
    # jump by the ratio is the fault ``strategy.split_check`` exists to catch.
    splits = np.zeros(n)
    if split_at is not None:
        splits[int(n * split_at)] = 2.0

    # Total return, exactly as CRSP defines it: on the ex-date the holder's wealth grows by
    # (close_t + dividend_t) / close_{t-1}. The reconstruction in strategy.py inverts this
    # identity, so on a clean tape the two must agree to floating point.
    tr = np.empty(n)
    tr[0] = price[0]
    for i in range(1, n):
        tr[i] = tr[i - 1] * (price[i] + div[i]) / price[i - 1]

    idx = pd.DatetimeIndex(dates, name="date")
    daily_tr = pd.DataFrame({"close": tr}, index=idx)
    daily_raw = pd.DataFrame({"close": price, "adj_close": tr / tr[0] * price[0],
                              "dividends": div, "stock_splits": splits}, index=idx)
    # Stamp the weekly and monthly bars the way the provider does — a weekly bar labelled
    # with the week's MONDAY carrying that week's Friday close, a monthly bar labelled with
    # the first of the month — so the audit is exercised against the real convention rather
    # than against a tidier one invented here.
    wk = daily_tr["close"].resample("W").last().dropna()
    wk.index = wk.index.to_period("W").to_timestamp()          # Monday of each week
    mo = daily_tr["close"].resample("ME").last().dropna()
    mo.index = mo.index.to_period("M").to_timestamp()          # first of each month
    weekly = pd.DataFrame({"close": wk})
    monthly = pd.DataFrame({"close": mo})
    frames = {"daily_tr": daily_tr, "daily_raw": daily_raw, "weekly": weekly,
              "monthly": monthly}
    truth = {"n_days": n, "n_years": n_years, "seed": seed, "dividend_yield": dividend_yield,
             "n_dividends": int((div > 0).sum()), "split_index": int(n * split_at)
             if split_at else None, "vol_ann": vol_ann, "drift_ann": drift_ann}
    return frames, truth


def corrupt_tape(frames: dict, drop_session: int | None = 500,
                 unapplied_split: bool = True, drop_dividend: int | None = 2,
                 seed: int = 971) -> tuple[dict, dict]:
    """Plant three specific, realistic faults into a known-good tape.

    - ``drop_session`` — one daily bar disappears (the classic missing-session fault, which is
      invisible unless you compare against an exchange calendar or against the weekly tape);
    - ``unapplied_split`` — the weekly series keeps the *pre-split* level after the split date,
      the fault that makes a 2:1 split look like a −50% day;
    - ``drop_dividend`` — one dividend is missing from the events column, so a
      price-plus-dividends reconstruction no longer reproduces the total-return series.

    Returns the corrupted frames and a description of exactly what was done, so a test can
    check that each audit fires for the right reason.
    """
    out = {k: v.copy() for k, v in frames.items()}
    planted = {}
    if drop_session is not None:
        d = out["daily_tr"].index[drop_session]
        out["daily_tr"] = out["daily_tr"].drop(index=d)
        out["daily_raw"] = out["daily_raw"].drop(index=d)
        planted["dropped_session"] = str(d.date())
    if unapplied_split:
        k = out["daily_raw"]["stock_splits"].to_numpy().argmax()
        split_date = out["daily_raw"].index[k]
        mask = out["weekly"].index >= split_date
        out["weekly"].loc[mask, "close"] = out["weekly"].loc[mask, "close"] * 2.0
        planted["unapplied_split_from"] = str(split_date.date())
    if drop_dividend is not None:
        pos = np.flatnonzero(out["daily_raw"]["dividends"].to_numpy() > 0)
        if len(pos) > drop_dividend:
            d = out["daily_raw"].index[pos[drop_dividend]]
            out["daily_raw"].loc[d, "dividends"] = 0.0
            planted["dropped_dividend"] = str(d.date())
    return out, planted


def synthetic_panel(n_assets: int = 3, n_years: int = 10, signal_strength: float = 1.0,
                    seed: int = 971, cash_rate_ann: float = 0.02):
    """Panel wrapper so the shared data-layer tests apply unchanged."""
    cols = {}
    for i in range(n_assets):
        frames, _ = synthetic_tape(n_years=n_years, seed=seed + i,
                                   dividend_yield=0.02 * float(signal_strength))
        cols[f"A{i}"] = frames["daily_tr"]["close"]
    prices = pd.DataFrame(cols)
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = pd.Series(np.cumprod(np.full(len(prices), cash_daily)), index=prices.index,
                     name="cash")
    truth = {"n_assets": n_assets, "n_years": n_years, "n_days": len(prices), "seed": seed,
             "signal_strength": float(signal_strength),
             "alpha_vol_eff": 0.05 * float(signal_strength),
             "cash_rate_ann": cash_rate_ann}
    return prices, cash, truth


def synthetic_daily(n_years: int = 10, signal_strength: float = 1.0, seed: int = 971):
    """Single-name convenience wrapper, matching the desk's shared shape."""
    prices, cash, truth = synthetic_panel(n_assets=1, n_years=n_years,
                                          signal_strength=signal_strength, seed=seed)
    return pd.DataFrame({"asset": prices.iloc[:, 0], "cash": cash}), truth

