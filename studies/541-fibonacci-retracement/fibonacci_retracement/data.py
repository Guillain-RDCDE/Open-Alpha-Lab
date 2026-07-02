"""Data layer for Study 541 — Fibonacci-Retracement.

The chartist claim tested here is narrow and mechanical: *when price retraces to a Fibonacci
fraction (0.236 / 0.382 / 0.5 / 0.618) of the prior swing, it tends to reverse there*. The placebo
is the identical machinery run on **arbitrary, non-Fibonacci** retracement fractions interleaved in
the same depth band (0.31 / 0.44 / 0.56). If Fibonacci levels are special, the Fib arm must beat
the placebo arm; if any round-ish level is as good, the two arms match.

Three generators:

* ``load_real`` — real daily OHLC for SPY + a basket of broad indices/ETFs (yfinance, no key),
  cache-first into ``_cache/`` so the reproducible core and the notebooks run offline once cached.
  Daily bars across decades give hundreds-to-thousands of swings — far more power than intraday.

* ``synthetic_tape`` — a *deterministic, offline* daily price path built as a clean alternating
  ZigZag. It is the **null price path** used to prove the *whole* pipeline (ZigZag → retracement →
  bin → two-sample test) fires **no false edge** on a genuine series with no planted level-effect
  (``fib_pull = 0``). (It also carries a ``fib_pull`` knob that scales the continuation after a
  Fib-depth pullback, but daily-noise pivot detection makes it a weak channel; the crisp positive
  control is ``synthetic_swings`` below.)

* ``synthetic_swings`` — the **positive control** generator: a clean panel of already-detected
  swings (a retracement fraction and a forward return per row) where swings at a Fibonacci depth
  are *given* an extra ``fib_pull`` of forward return. This directly exercises the binning +
  two-sample edge test, which must recover the planted effect (and stay flat at ``fib_pull = 0``).

Survivorship / data note: these are broad *price-index / ETF* tapes (survivors, but no
cross-sectional single-name basket), auto-adjusted total-return daily closes. There is no
single-name delisting bias here; the tape-availability limit is that yfinance is the only free
source, named on the SIGNAL axis.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# SPY = the headline tape; the rest are deep-history broad indices/ETFs chartists draw
# retracements on. All survivors — price-index tapes, so no cross-sectional survivorship sort.
TICKERS = ("SPY", "QQQ", "DIA", "IWM", "^GSPC", "^IXIC", "^DJI", "GLD")

# The canonical Fibonacci retracement levels vs a matched placebo of arbitrary NON-Fibonacci
# fractions INTERLEAVED within the same 0.236..0.618 depth span (so the two arms sample the same
# range of pullback depths — only the exact fractions differ) and non-overlapping at a +/-0.025
# tolerance: 0.31 / 0.44 / 0.56.
FIB_LEVELS = (0.236, 0.382, 0.5, 0.618)
PLACEBO_LEVELS = (0.31, 0.44, 0.56)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"fib_{safe}_1d.parquet")


def load_real(ticker: str = "SPY", period: str = "max",
              cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              retries: int = 3) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``; cache-first (network only on a cache miss or ``fetch``).

    Returns a tz-naive daily OHLC frame (open/high/low/close) indexed by date. The cache is a
    parquet under ``_cache/``; once present every re-run is offline. On a cache miss (or
    ``fetch=True``) it pulls from yfinance with a small retry/backoff and writes the parquet.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(ticker, period=period, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars.dropna().sort_index()


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def load_basket(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                fetch: bool = False) -> dict[str, pd.DataFrame]:
    """Load every cached ticker we can (cache-first). Returns {ticker: OHLC frame}."""
    out = {}
    for tk in tickers:
        if fetch or have_real(tk, cache_dir):
            try:
                out[tk] = load_real(tk, cache_dir=cache_dir, fetch=fetch)
            except Exception:
                continue
    return out


# --------------------------------------------------------------------------- #
# Synthetic price-path NULL — deterministic alternating-ZigZag tape (offline pipeline proof)
# --------------------------------------------------------------------------- #
def synthetic_tape(n_swings: int = 900, fib_pull: float = 0.0, seed: int = 541,
                   daily_vol: float = 0.004, start: str = "1998-01-05"
                   ) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape built as a clean alternating ZigZag (the NULL price path).

    Primary role: prove the whole pipeline (ZigZag -> retracement -> bin -> two-sample test) finds
    **no false edge** on a genuine price series at ``fib_pull = 0``. A ``fib_pull`` knob is also
    provided (it scales the continuation leg after a Fib-depth pullback), but on daily bars the
    noise in pivot detection makes it a weak detection channel — the crisp positive control is
    ``synthetic_swings``.

    The tape is a piecewise path through an **alternating pivot sequence** (up-leg, down-leg,
    up-leg, ...) so that the ZigZag detector recovers exactly the planted pivots — which is the only
    way a plant keyed to retracement *depth* can be seen by a retracement-depth detector.

    Each *reference* leg has a base size ``leg_size`` (in log space). The next, opposing leg is a
    **retracement** of a fraction ``r`` of it (``r`` drawn uniformly on [0.20, 0.90] in BOTH worlds,
    so Fib and placebo depths appear equally often). After that retracement pivot, the following leg
    is the **continuation** — the chartist's "does the trend resume?" question:

      - **null** (``fib_pull = 0``) — the continuation leg's size is a fair, depth-independent draw
        (it may extend past the prior extreme or not, with no dependence on ``r``): nothing special
        at any depth.
      - **planted** (``fib_pull > 0``) — *only when the retracement landed near a Fibonacci depth*
        does the continuation leg get a large upward push of ``(1 + fib_pull)`` × the base size (a
        strong, reliable resumption). Off-Fib retracements get the plain base leg. That is exactly
        "price reverses / the trend resumes *at* the Fib level, and nowhere else".

    Because the leg *sizes* are what carries the plant (not the leg count), the number of pivots and
    the depth distribution are identical across ``fib_pull`` — only the forward move after a Fib-depth
    retracement differs, which is precisely what the detector should (and does) pick up.

    Returns ``(bars, truth)``.
    """
    rng = np.random.default_rng(seed)
    fibs = np.array(FIB_LEVELS)
    leg_size = 0.30                       # base reference leg in log space; even r=0.20 retraces >5%
    days_per_leg = 18                     # a 20-day forward window captures ~one continuation leg

    # Build the alternating pivot log-price sequence as (reference leg, retracement leg) pairs.
    # For each swing: emit a reference leg of magnitude `ref`, then a retracement leg of `r*ref`
    # in the opposite direction. The fraction `r` is what the detector reads as the retracement of
    # the reference swing (P0->P1->P2 => retr == r by construction). The NEXT swing's reference leg
    # (the "continuation") is enlarged ONLY when this `r` landed near a Fib depth in the planted
    # world — so the forward move after a Fib-depth pullback is bigger, exactly the chartist claim.
    logp = [0.0]
    sign = 1.0
    ref = leg_size
    for _ in range(n_swings):
        # reference leg (this is also the "continuation" the previous swing set up)
        logp.append(logp[-1] + sign * ref)
        sign = -sign
        # retracement leg of fraction r of the reference leg
        r = float(rng.uniform(0.20, 0.90))
        logp.append(logp[-1] + sign * ref * r)
        sign = -sign
        # size the NEXT reference leg (the continuation): the SAME base size everywhere EXCEPT when
        # this pullback landed near a Fib depth in the planted world, where the resumption is
        # stronger. Keeping the off-Fib leg at the fixed base size makes the pivot structure (and
        # the retracement-depth distribution) essentially identical across `fib_pull` and across the
        # two arms — so the ONLY thing that moves the Fib-vs-placebo edge is the planted effect.
        near_fib = bool(np.min(np.abs(fibs - r)) <= 0.03)
        if fib_pull > 0.0 and near_fib:
            ref = leg_size * (1.0 + fib_pull)                    # strong resumption at a Fib depth
        else:
            ref = leg_size                                       # fixed base leg everywhere else

    pivots_log = np.asarray(logp)

    # Interpolate a daily path between pivots with small noise; each leg spans `days_per_leg` bars.
    seg_rets = []
    for a, b in zip(pivots_log[:-1], pivots_log[1:]):
        step = (b - a) / days_per_leg
        seg_rets.extend((step + rng.normal(0.0, daily_vol, days_per_leg)).tolist())
    eps = np.asarray(seg_rets, dtype=float)
    n_days = len(eps)
    cal = pd.bdate_range(start=start, periods=n_days)
    close = 100.0 * np.exp(np.cumsum(eps))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close},
                        index=pd.DatetimeIndex(cal, name="date"))
    truth = {"fib_pull": fib_pull, "n_swings": n_swings, "seed": seed,
             "daily_vol": daily_vol, "has_effect": fib_pull != 0.0}
    return bars, truth


# --------------------------------------------------------------------------- #
# Synthetic SWING PANEL — the clean positive-control generator
# --------------------------------------------------------------------------- #
def synthetic_swings(n: int = 1500, fib_pull: float = 0.0, seed: int = 541,
                     ret_vol: float = 0.06) -> tuple[pd.DataFrame, dict]:
    """A deterministic panel of *already-detected* swings, with a planted Fibonacci level-effect.

    This is the positive control for the **inference** stage (binning + the Fib-minus-placebo
    two-sample *t*). Each row is one swing that the ZigZag would have handed us: a realised
    retracement fraction ``retr`` (drawn uniformly on [0.20, 0.90], so Fibonacci and placebo depths
    occur equally often), a resume direction ``dir`` (always the tradable "+1", the trend-resume
    bet), and the **forward return** of the reversal bet:

        forward_ret = drift + (fib_pull if retr is near a Fibonacci depth else 0) + noise

    - ``fib_pull = 0`` → the null: the forward return is unrelated to whether the retracement
      landed on a Fibonacci level, so the Fib arm and the placebo arm must be statistically
      indistinguishable (edge *t* ≈ 0).
    - ``fib_pull > 0`` → the planted world: swings that retraced to a Fibonacci depth earn an extra
      ``fib_pull`` of forward return — exactly "price reverses harder at the Fib level". The Fib arm
      must then beat the placebo arm (edge *t* > 0), growing with ``fib_pull``.

    Separating this clean swing panel from the ``synthetic_tape`` price path is deliberate: the tape
    validates that the *whole* pipeline (ZigZag → retracement → bin → test) fires **no false edge**
    on a genuine price series (the null), while this panel validates that the binning + two-sample
    test **catch a planted level-effect** when one exists. Together they bound the engine on both
    sides. The ``sig_idx`` here is a synthetic bar index used only to feed ``run_trades``-style
    consumers a per-swing forward return; the number is the ``forward_ret`` itself.

    Returns ``(swings, truth)`` where ``swings`` has columns ``retr``, ``dir``, ``forward_ret``.
    """
    rng = np.random.default_rng(seed)
    fibs = np.array(FIB_LEVELS)
    retr = rng.uniform(0.20, 0.90, size=n)
    near_fib = np.min(np.abs(fibs[None, :] - retr[:, None]), axis=1) <= 0.03
    drift = 0.03
    forward = drift + np.where(near_fib & (fib_pull > 0.0), fib_pull, 0.0) \
        + ret_vol * rng.standard_normal(n)
    swings = pd.DataFrame({
        "retr": retr.astype(float),
        "dir": np.ones(n, dtype=int),
        "forward_ret": forward.astype(float),
    })
    truth = {"fib_pull": fib_pull, "n": n, "seed": seed, "has_effect": fib_pull != 0.0}
    return swings, truth


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
