"""Data layer for Study 536 (Anomaly-Decay-Post-Publication).

This is a *methodology* study, so the data layer supplies two tapes of the SAME
shape — a monthly total-return panel of a fixed survivor basket — so the
publication-decay reckoning can run identically on synthetic and real data:

- ``synthetic_panel`` — a *deterministic, offline* generator. It plants a known
  cross-sectional anomaly that is STRONG before a "publication" month and DECAYS
  by a known fraction afterwards. The engine must recover that planted decay (the
  positive control) — and at ``post_decay = 1.0`` the post edge vanishes entirely
  (the null), so the demo can prove the split-at-publication machinery is faithful.

- ``load_real`` — a real monthly total-return panel of a fixed ~40-name large-cap
  **survivor basket**, cache-first parquet under THIS study's ``_cache/`` with a
  guarded yfinance pull. Each classic anomaly (momentum, low-vol, short-term
  reversal, long-term reversal) is a price-only cross-sectional signal with a real
  academic **publication year**; we split each name's sample at that year and
  compare the long-short before vs after.

Survivorship: the basket is names still trading in 2026. That biases the *level* of
every anomaly (survivors are the winners), but the study's quantity of interest is
the *within-anomaly pre/post ratio*, which the survivorship tilt enters on both
sides. We name it on the Signal axis anyway and add an opt-in guard note.

One execution lag: a signal formed from data through the close of month *t* trades
the *month t+1* total return (``shift`` applied once in ``forward_returns``). No
look-ahead, no same-bar fills.

Timestamp note: monthly indices are built with ``pd.period_range`` (never a long
``pd.date_range``) to dodge the ns-Timestamp overflow on CI pandas.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

MONTHS_PER_YEAR = 12

# A fixed ~40-name large-cap SURVIVOR basket (still trading in 2026), the same kind
# of basket Studies 238 / 330 / 363 use. Survivorship is acknowledged on the Signal
# axis; it biases anomaly *levels*, not the within-anomaly pre/post *ratio* we test.
BASKET = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "PG", "KO",
    "XOM", "CVX", "WMT", "HD", "DIS", "INTC", "CSCO", "PFE", "MRK", "VZ",
    "T", "BA", "CAT", "MMM", "GE", "IBM", "ORCL", "QCOM", "TXN", "HON",
    "UNH", "ABT", "MCD", "NKE", "COST", "PEP", "WFC", "C", "BAC", "GS",
)

# The classic anomalies we tear down, each with a REAL publication year. The
# publication year is the split point: McLean & Pontiff (2016) report anomaly
# returns roughly halve in the years AFTER the defining paper appears.
#   - mom12_1      Jegadeesh & Titman (1993) — 12-1 momentum
#   - lowvol       Ang, Hodrick, Xing & Zhang (2006) — low idiosyncratic/total vol
#   - st_reversal  Jegadeesh (1990); Lehmann (1990) — 1-month short-term reversal
#   - lt_reversal  De Bondt & Thaler (1985) — 3-year long-term reversal
ANOMALIES = {
    "mom12_1":     {"pub_year": 1993, "label": "12-1 momentum",        "side": +1},
    "lowvol":      {"pub_year": 2006, "label": "low volatility",       "side": +1},
    "st_reversal": {"pub_year": 1990, "label": "1-month reversal",     "side": +1},
    "lt_reversal": {"pub_year": 1985, "label": "3-year reversal",      "side": +1},
}


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 40,
    n_months: int = 600,
    pub_month: int = 300,
    pre_edge: float = 0.010,
    post_decay: float = 0.5,
    vol: float = 0.06,
    seed: int = 536,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """A reproducible monthly panel with a planted anomaly that DECAYS at ``pub_month``.

    Returns ``(rets, signal, pub_month)``:
      - ``rets``  (n_months x n_names) monthly total returns,
      - ``signal`` (n_months x n_names) a cross-sectional score known at month-end,
      - ``pub_month`` the integer index where the anomaly's strength steps down.

    The data-generating process: each name has i.i.d. noise plus an anomaly premium
    proportional to its (de-meaned) signal. The premium is ``pre_edge`` per unit
    signal BEFORE ``pub_month`` and ``pre_edge * (1 - post_decay)`` after — so:
      - ``post_decay = 0.5`` halves the edge (the canonical McLean-Pontiff control),
      - ``post_decay = 1.0`` kills it entirely (the null — post edge is pure noise),
      - ``post_decay = 0.0`` leaves it intact (a no-decay control).

    The signal is the SAME object the strategy will sort on, so a faithful engine
    must recover the planted pre/post step exactly.
    """
    rng = np.random.default_rng(seed)
    names = [f"n{i:02d}" for i in range(n_names)]

    # A persistent cross-sectional characteristic each month (the anomaly signal).
    signal = rng.normal(0.0, 1.0, size=(n_months, n_names))
    # cross-sectionally de-mean so the long-short is dollar-neutral by construction
    signal = signal - signal.mean(axis=1, keepdims=True)

    noise = rng.normal(0.0, vol, size=(n_months, n_names))
    premium = np.full(n_months, pre_edge)
    premium[pub_month:] = pre_edge * (1.0 - post_decay)
    # The signal known at month-end *t* predicts month *t+1*'s return (one lag), so we
    # apply the premium to the PREVIOUS month's signal — matching the strategy layer,
    # which sorts on signal_t and earns rets_{t+1}.
    sig_lag = np.vstack([np.zeros((1, n_names)), signal[:-1]])
    rets = noise + premium[:, None] * sig_lag

    idx = pd.period_range("1970-01", periods=n_months, freq="M")
    rets_df = pd.DataFrame(rets, index=idx, columns=names)
    sig_df = pd.DataFrame(signal, index=idx, columns=names)
    return rets_df, sig_df, pub_month


# ---------------------------------------------------------------------------
# Real panel — survivor basket, cache-first
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, tag: str = "basket") -> str:
    return os.path.join(cache_dir, f"{tag}_monthly.parquet")


def load_real(
    start: str = "1980-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    basket: tuple[str, ...] = BASKET,
) -> pd.DataFrame:
    """Real monthly total-return panel of the survivor basket → a (T x N) frame.

    Cache-first: reads the parquet under THIS study's ``_cache/`` and never touches
    the network unless ``fetch=True``. On a cache miss with ``fetch=False`` it raises
    ``FileNotFoundError`` (the offline contract) so the reproducible core stays
    deterministic.

    Columns are tickers; the index is a monthly ``PeriodIndex``. Each cell is the
    month's total return (auto-adjusted close-to-close). Names that had not yet
    listed simply carry NaN early — the cross-sectional sort drops them month by
    month, so the panel is unbalanced but honest.

    Survivorship: every name is a 2026 survivor. Named on the Signal axis; the
    pre/post *ratio* (not the level) is the quantity of interest, so the bias enters
    both sides.
    """
    cpath = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(cpath):
            raise FileNotFoundError(
                f"No cached panel at {cpath}. "
                f"Call load_real(fetch=True) once to populate (needs network)."
            )
        panel = pd.read_parquet(cpath)
    else:
        panel = _fetch_panel(basket, start)
        os.makedirs(cache_dir, exist_ok=True)
        panel.to_parquet(cpath)

    # Restore the PeriodIndex (parquet round-trips it as a PeriodArray or timestamps).
    if not isinstance(panel.index, pd.PeriodIndex):
        panel.index = pd.PeriodIndex(panel.index, freq="M")
    # Drop the in-progress final month so a stamped run never includes a partial bar.
    today = pd.Timestamp.today().to_period("M")
    panel = panel[panel.index < today]
    return panel.sort_index()


def _fetch_one(ticker: str, start: str, retries: int = 3) -> pd.Series:
    """Daily auto-adjusted closes for one ticker, with retries (yfinance flakiness)."""
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker, start=start, auto_adjust=True, progress=False, threads=False
            )
            if raw is not None and not raw.empty:
                s = raw["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                return s.dropna()
        except Exception as e:  # network / parse hiccup
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"no price data for {ticker} after {retries} tries ({last_err})")


def _fetch_panel(basket: tuple[str, ...], start: str) -> pd.DataFrame:
    """Build the monthly total-return panel from daily closes (one column per name)."""
    monthly = {}
    for tk in basket:
        try:
            px = _fetch_one(tk, start)
        except Exception as e:
            print(f"  [skip {tk}: {e}]")
            continue
        m_close = px.resample("ME").last()
        monthly[tk] = m_close.pct_change()
    if not monthly:
        raise RuntimeError("no tickers fetched — network down?")
    df = pd.DataFrame(monthly)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df.dropna(how="all")


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the panel, for the as-of stamp."""
    arr = np.ascontiguousarray(np.nan_to_num(panel.to_numpy()).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
