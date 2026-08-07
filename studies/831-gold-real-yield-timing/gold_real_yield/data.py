"""Data layer for Study 831 — Gold Real-Yield Timing.

Gold is famously said to track **real yields inversely**: when the real (inflation-
protected) yield falls, gold rises, and vice versa. The tradeable twist this study
tests is whether the **real-yield trend** *predicts* forward gold — i.e. whether a
"real yields are falling → own gold" timing rule beats simply buying and holding
gold, net of costs.

The official 10-year real yield (the TIPS constant-maturity series, FRED ``DFII10``)
is not a no-key Yahoo ticker, so this study builds a **retail-reachable real-yield
proxy** from what yfinance *does* expose:

* **Primary gauge — TIP total return as an inverse real-yield meter.** The iShares
  TIPS ETF (``TIP``) is a basket of inflation-protected Treasuries; its (total-return,
  ``auto_adjust``) price rises exactly when real yields fall. So the trailing return of
  TIP is a clean, sign-flipped proxy for the *change* in the real yield:

      real_yield_fall_t(L)  =  log(TIP_t) − log(TIP_{t−L})      (>0  ⇔  real yields fell)

  A positive value means "real yields have been falling" — the claim's *buy gold*
  signal. Duration and offset constants drop out because the signal is **ranked**, so
  no arbitrary scale enters the sort.

* **Secondary gauge — nominal minus breakeven (``TNX − BEI``).** The other textbook
  identity, real ≈ nominal − breakeven-inflation. We proxy the 10y breakeven by the
  relative total return of TIPS vs nominal Treasuries (``IEF``): when breakevens rise,
  TIPS out-earn nominals, so ``bei ∝ log(TIP) − log(IEF)``. The proxy real-yield level
  ``ry = TNX − 100·(log TIP − log IEF)`` is used only as a robustness cross-check
  (does a second, independent proxy agree with the TIP gauge?); the headline rides the
  TIP gauge alone.

Both proxies are **model simplifications, named openly on the SIGNAL axis** — the
reason a `REAL` certification is judged strictly against a robust HAC *t* on the real
gold tape, not against the proxy's plausibility.

Two tapes, one schema (a tz-naive daily frame ``[GLD_close, TIP_close, IEF_close,
TNX, ry, GLD_ret]``):

* ``synthetic_daily`` — a deterministic, offline generator. A ``link_beta`` knob plants
  the *contemporaneous* inverse gold↔real-yield link; a separate ``edge`` knob plants a
  genuine *predictive* timing edge (forward gold loads on the lagged real-yield-fall
  rank). ``edge = 0`` is the null world — the inverse link can be present while the
  *timing* signal carries nothing, exactly the case the desk expects on the real tape.
  Tests never touch the network.
* ``fetch_daily`` — the real yfinance tape (GLD + TIP + IEF + ^TNX daily closes),
  cache-first into this study's own ``_cache/`` (retry up to 4×) so the reproducible
  core never needs the network. ``load_series`` / ``load_panel`` read it OFFLINE.

No look-ahead: the real-yield trend measured at the close of day *t* forms a signal
that trades at the close of day *t+1* (a one-bar execution lag, applied identically on
both tapes and across the robustness sweep — see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers: the gold ETF, the TIPS ETF (real gauge), a nominal-Treasury ETF
# (for the breakeven cross-check), and the 10-year nominal yield.
TICKERS = ["GLD", "TIP", "IEF", "^TNX"]

START = "2004-01-01"        # GLD inception is 2004-11; the dropna trims to it
AS_OF = "2026-06-30"        # last complete calendar month at publication

# Breakeven scaling for the SECONDARY proxy real-yield LEVEL (points). The level is a
# cross-check only (headline ranks the TIP gauge), so this constant is cosmetic.
BEI_SCALE = 100.0

__all__ = [
    "TICKERS", "START", "AS_OF", "DEFAULT_CACHE",
    "fetch", "fetch_daily", "have_real", "load_series", "load_panel",
    "real_yield_proxy", "synthetic_daily", "fingerprint",
]


# ---------------------------------------------------------------------------
# The real-yield proxy (shared by both tapes)
# ---------------------------------------------------------------------------
def real_yield_proxy(tip: pd.Series, ief: pd.Series, tnx: pd.Series,
                     scale: float = BEI_SCALE) -> pd.Series:
    """Secondary real-yield-LEVEL proxy ``ry = TNX − scale·(log TIP − log IEF)`` (points).

    ``log TIP − log IEF`` is a breakeven-inflation proxy (TIPS out-earn nominals when
    breakevens rise); subtracting it from the nominal 10y yield leaves a real-yield
    stand-in. Causal (only trailing prices), so the value at day *t* is public at the
    close of *t*. Used as a robustness cross-check against the TIP gauge; the additive
    offset is arbitrary (a level proxy), so only its *changes* carry meaning.
    """
    bei = scale * (np.log(tip) - np.log(ief))
    return (tnx - bei).rename("ry")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 5000,
    edge: float = 0.0,
    link_beta: float = 8.0,
    seed: int = 831,
    gld_vol: float = 0.010,
    start: str = "2004-11-18",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of real-yield timing power.

    Structure:

    * ``ry`` (the true real yield, %) is an AR(1) around ~0.5%; ``bei`` (breakeven, %)
      is a smoother AR(1) around ~2%; the nominal 10y ``TNX = ry + bei``.
    * ``TIP`` and ``IEF`` are duration-priced off the real and nominal yields
      (``price ∝ exp(−dur·Δyield)``), so TIP rises exactly when the real yield falls —
      the real inverse-gauge mechanism.
    * The **real-yield-fall** signal ``ryfall = log(TIP_t) − log(TIP_{t−63})`` is ranked
      out-of-sample; forward gold loads on it with two independent knobs::

          GLD_ret_t = −link_beta·Δ(ry_t)/100        (contemporaneous inverse link)
                      + edge·(ryfall_rank_{t−1} − ½) (lagged PREDICTIVE timing edge)
                      + eps_t

      ``link_beta > 0`` makes gold move inversely with same-day real-yield changes (the
      famous fact); ``edge = 0`` is the null for the *timing* claim (the trend predicts
      nothing forward); ``edge > 0`` plants a genuine timing edge.

    Returns ``(df, truth)`` with columns ``[GLD_close, TIP_close, IEF_close, TNX, ry,
    GLD_ret]``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # True real yield: AR(1) around 0.5%
    ry = np.empty(n_days)
    ry[0] = 0.5
    phi_ry, sig_ry, mu_ry = 0.995, 0.03, 0.5
    for i in range(1, n_days):
        ry[i] = phi_ry * ry[i - 1] + (1 - phi_ry) * mu_ry + rng.normal(0, sig_ry)
    ry = np.clip(ry, -2.0, 4.5)

    # Breakeven inflation: smoother AR(1) around 2%
    bei = np.empty(n_days)
    bei[0] = 2.0
    phi_b, sig_b, mu_b = 0.997, 0.02, 2.0
    for i in range(1, n_days):
        bei[i] = phi_b * bei[i - 1] + (1 - phi_b) * mu_b + rng.normal(0, sig_b)
    bei = np.clip(bei, 0.2, 4.0)

    tnx = ry + bei                                   # nominal 10y yield (%)
    d_ry = np.concatenate([[0.0], np.diff(ry)])      # daily real-yield change (%)
    d_nom = np.concatenate([[0.0], np.diff(tnx)])    # daily nominal-yield change (%)

    dur_real, dur_nom = 7.5, 7.0
    tip_close = 100.0 * np.exp(-dur_real * np.cumsum(d_ry / 100.0)
                               + rng.normal(0, 0.0015, n_days))
    ief_close = 100.0 * np.exp(-dur_nom * np.cumsum(d_nom / 100.0)
                               + rng.normal(0, 0.0015, n_days))

    # Real-yield-fall momentum from the TIP gauge (63-day), ranked out-of-sample.
    log_tip = pd.Series(np.log(tip_close), index=idx)
    ryfall = log_tip - log_tip.shift(63)
    rank = ryfall.rolling(252, min_periods=63).rank(pct=True)

    eps = rng.normal(0.0, gld_vol, n_days)
    gld_ret = np.empty(n_days)
    for i in range(n_days):
        lag = rank.iloc[i - 1] if i >= 1 else np.nan
        predictive = 0.0 if np.isnan(lag) else edge * (lag - 0.5)
        gld_ret[i] = -link_beta * (d_ry[i] / 100.0) + predictive + eps[i]
    gld_close = 100.0 * np.exp(np.cumsum(gld_ret))

    ry_proxy = real_yield_proxy(
        pd.Series(tip_close, index=idx), pd.Series(ief_close, index=idx),
        pd.Series(tnx, index=idx),
    )
    df = pd.DataFrame(
        {
            "GLD_close": gld_close,
            "TIP_close": tip_close,
            "IEF_close": ief_close,
            "TNX": tnx,
            "ry": ry_proxy.to_numpy(),
            "GLD_ret": gld_ret,
        },
        index=idx,
    )
    df.index.name = "date"
    truth = {"edge": edge, "link_beta": link_beta, "gld_vol": gld_vol,
             "n_days": n_days, "seed": seed, "start": start}
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily closes, cache-first
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "daily_gold_real_yield.parquet")


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(cache_dir))


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = START,
    asof: str = AS_OF,
    retries: int = 4,
) -> pd.DataFrame:
    """Real daily GLD / TIP / IEF / 10y-yield tape with the real-yield proxy; cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else
    raises ``FileNotFoundError``. Network is touched only on an explicit ``fetch=True``
    (retried up to ``retries`` times with a short back-off, then cached). Returned frame:
    ``[GLD_close, TIP_close, IEF_close, TNX, ry, GLD_ret]`` with a tz-naive
    ``DatetimeIndex`` named ``date``, sliced to ``[start, asof]`` (partial current month
    dropped).
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape at {path}. Call fetch_daily(fetch=True) once to populate."
            )
        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    import yfinance as yf  # lazy

    last_err: Exception | None = None
    raw = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                TICKERS, start=start, end=asof, interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.5 * attempt)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no daily bars after {retries} tries ({last_err})")

    closes = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].copy()
    closes.columns = [str(c).replace("^", "") for c in closes.columns]
    closes.index.name = "date"
    closes = closes.rename(columns={"GLD": "GLD_close", "TIP": "TIP_close", "IEF": "IEF_close"})
    for c in ("GLD_close", "TIP_close", "IEF_close", "TNX"):
        closes[c] = closes[c].ffill(limit=5)
    closes = closes.dropna(subset=["GLD_close", "TIP_close", "IEF_close", "TNX"])

    closes["ry"] = real_yield_proxy(closes["TIP_close"], closes["IEF_close"], closes["TNX"]).to_numpy()
    closes["GLD_ret"] = np.log(closes["GLD_close"]).diff()
    closes = closes.dropna(subset=["GLD_ret"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    closes = closes[(closes.index >= lo) & (closes.index <= hi)]
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)

    os.makedirs(cache_dir, exist_ok=True)
    closes.to_parquet(path)
    return closes


def fetch(start: str = START) -> None:
    """Download the real tape through :func:`fetch_daily` and cache it (network; once)."""
    fetch_daily(fetch=True, start=start)


def load_series(cache_dir: str = DEFAULT_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily tape, OFFLINE (no yfinance import), sliced to ``[START, asof]``."""
    df = fetch_daily(fetch=False, cache_dir=cache_dir)
    hi = pd.Timestamp(asof)
    return df[df.index <= hi]


# convenience alias mirroring the desk's panel loaders
load_panel = load_series


def fingerprint(df: pd.DataFrame, col: str = "GLD_close") -> str:
    """A short content fingerprint of a daily tape column, for the as-of stamp."""
    s = df[col] if col in df.columns else df.iloc[:, 0]
    h = hashlib.sha1(np.ascontiguousarray(s.fillna(0).to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
