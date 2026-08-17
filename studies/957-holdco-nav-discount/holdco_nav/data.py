"""Data layer for Study 957 — Holdco Discount (listed holdcos vs their listed stakes).

A holding company whose value is dominated by *one listed stake* is the rare case where
net asset value is visible on the tape: multiply the stake's share price by the number of
shares the holdco owns, divide by the holdco's own share count, and you have a NAV per
share you can mark every evening. The gap between that NAV and the holdco's own share
price is the **holdco discount**.

Two tapes, one shape:

- ``fetch`` / ``load_prices`` / ``load_price_only`` — daily closes from Yahoo! Finance
  (``yfinance``). We keep **both** conventions, because the two are used for different
  jobs and conflating them is the classic error in this corner:

  * ``prices_<TKR>_1d.parquet`` — ``auto_adjust=True`` **total-return** closes. These are
    what an owner earns, so they drive every *return* in the backtest.
  * ``pxonly_<TKR>_1d.parquet`` — ``auto_adjust=False`` ``Close``, i.e. **price-only**
    (split-adjusted, *not* dividend-adjusted). These are what a *valuation* is made of, so
    they drive the NAV ratio. Using total-return closes on both legs of a NAV ratio would
    quietly inject the two names' dividend-yield difference into the "discount".

  ``fetch`` touches the network and caches parquet under the shared ``studies/_cache``
  (retry up to 4x); ``load_*`` read that cache **offline** and never import yfinance. The
  whole test-suite runs with NO cache present (synthetic only), so CI is green on a fresh
  checkout.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator. A panel
  of (stake, holdco) pairs where the stake is a GBM and the log-discount is an
  Ornstein-Uhlenbeck process whose mean-reversion speed is scaled by ``signal_strength``:
  at ``signal_strength=1`` the discount genuinely mean-reverts and a buy-the-wide-discount
  rule must pay; at ``signal_strength=0`` the discount is a driftless random walk and the
  rule must earn nothing. Seeds fixed -> tests are deterministic.

**The NAV proxy is an ASSUMPTION, and the study says so everywhere.** ``STAKES`` below
hardcodes, per holdco, the number of stake shares owned per holdco share (``k``) and a
per-share "everything else" constant (``other_per_share``: unlisted subsidiaries minus
holding-company net debt). Both are frozen at a recent snapshot; in reality both drift.
The consequence is precise and worth stating plainly: the *level* of each discount series
is only as good as this table, so the study's tests are built on the *trailing-standardised*
discount (a z-score against each name's own preceding two years), which a constant error in
``k`` or ``other_per_share`` barely moves. ``strategy.calibration_sweep`` perturbs both.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# The panel — seven holdcos whose single dominant asset is a listed stake.
# --------------------------------------------------------------------------- #
# Every field below the tickers is a PROXY / ASSUMPTION, frozen at a 2026 snapshot:
#
#   k                — stake shares owned per holdco share (a ratio, so share splits on
#                      either leg cancel against the split-adjusted price series). For the
#                      three names whose stake arithmetic is unambiguous (Heineken Holding,
#                      Christian Dior, Liberty Broadband) this is the published share count.
#                      For the four whose ADR ratios and chained holdings make the arithmetic
#                      opaque (Naspers, Prosus, SoftBank, Bollore) it is *anchored*: chosen so
#                      the discount on ``anchor_date`` equals the widely reported
#                      ``anchor_disc``. A single anchor per name, stated openly, beats a
#                      spurious-precision share count we cannot verify from the tape.
#   other_per_share  — unlisted assets minus holding-company net debt, per holdco share,
#                      in the pair's own currency. Zero wherever the stake is
#                      overwhelmingly the whole company.
#   stake_share_nav  — our rough estimate of how much of NAV the listed stake is. Used
#                      only for disclosure, never in a calculation.
#   end              — a structural-break truncation, chosen on *corporate events*, never
#                      on performance (documented per name in docs/references.md).
#
# Both legs of every pair quote in the SAME currency, so no FX conversion enters the ratio.
STAKES = {
    "HEINEKEN_HOLDING": dict(
        holdco="HEIO.AS", stake="HEIA.AS", k=0.9995, other_per_share=0.0,
        stake_share_nav=1.00, currency="EUR", start="2004-05-04", end=None,
        note="Heineken Holding NV owns 50.005% of Heineken NV and essentially nothing else.",
    ),
    "CHRISTIAN_DIOR": dict(
        holdco="CDI.PA", stake="MC.PA", k=1.1745, other_per_share=0.0,
        stake_share_nav=0.97, currency="EUR", start="2004-01-02", end=None,
        note="Christian Dior SE owns ~42.4% of LVMH; the residual (Dior Couture) sits inside LVMH.",
    ),
    "LIBERTY_BROADBAND": dict(
        holdco="LBRDK", stake="CHTR", k=0.3161, other_per_share=-20.1,
        stake_share_nav=1.00, currency="USD", start="2014-11-05", end="2024-11-12",
        note="45.6m Charter shares over ~144m Liberty Broadband shares; GCI Alaska minus net debt "
             "is the negative 'other' term. TRUNCATED at 2024-11-12: Charter and Liberty "
             "Broadband signed a definitive all-stock merger agreement on 2024-11-13, so from "
             "that date the gap is a MERGER SPREAD with a contractual exchange ratio and a "
             "convergence date (Study 366's subject), not a holdco discount.",
    ),
    "NASPERS": dict(
        holdco="NPSNY", stake="TCEHY", k=0.30062, other_per_share=0.0,
        stake_share_nav=0.90, currency="USD", start="2010-01-05", end=None,
        anchor_date="2026-06-30", anchor_disc=0.40,
        note="Naspers' look-through Tencent stake, held via its ~73% of Prosus; ADR-to-ADR. "
             "k anchored to the widely reported ~40% discount at 2026-06-30.",
    ),
    "PROSUS": dict(
        holdco="PROSY", stake="TCEHY", k=0.22507, other_per_share=0.0,
        stake_share_nav=0.80, currency="USD", start="2020-02-24", end=None,
        anchor_date="2026-06-30", anchor_disc=0.30,
        note="Prosus NV holds Tencent directly; ADR-to-ADR. Overlaps NASPERS by construction. "
             "k anchored to the reported ~30% discount at 2026-06-30.",
    ),
    "BOLLORE": dict(
        holdco="BOL.PA", stake="VIV.PA", k=1.34292, other_per_share=0.0,
        stake_share_nav=0.60, currency="EUR", start="2004-01-02", end="2024-11-30",
        anchor_date="2024-11-29", anchor_disc=0.50,
        note="Bollore SE's ~27% of Vivendi. TRUNCATED at 2024-11-30: Vivendi's December 2024 "
             "split into four listed pieces destroys the observable-NAV identity.",
    ),
    "SOFTBANK": dict(
        holdco="SFTBY", stake="BABA", k=0.09203, other_per_share=0.0,
        stake_share_nav=0.65, currency="USD", start="2014-09-19", end="2021-12-31",
        anchor_date="2021-12-31", anchor_disc=0.45,
        note="SoftBank Group's Alibaba stake, ADR-to-ADR. TRUNCATED at 2021-12-31: SoftBank "
             "disposed of essentially the whole stake through 2022-2023 prepaid forwards.",
    ),
}

HOLDCOS = tuple(STAKES.keys())
CASH_TICKER = "IRX"          # ^IRX, the 13-week T-bill discount rate (a PROXY cash leg)
CASH_ETF = "BIL"             # the tradable cross-check over its own (2007->) window

TICKERS = tuple(sorted(
    {v["holdco"] for v in STAKES.values()} | {v["stake"] for v in STAKES.values()}
    | {"^IRX", "BIL"}
))


# --------------------------------------------------------------------------- #
# Real tape — cache-only by default
# --------------------------------------------------------------------------- #
def _safe(ticker: str) -> str:
    return ticker.replace("=", "").replace("^", "").replace("/", "")


def _cache_path(ticker: str, cache_dir: str, kind: str = "prices") -> str:
    return os.path.join(cache_dir, f"{kind}_{_safe(ticker)}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2003-06-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache **two** parquet files each.

    ``prices_<T>_1d.parquet`` carries the total-return close (``auto_adjust=True``) — the
    return convention. ``pxonly_<T>_1d.parquet`` carries the split-adjusted, *not*
    dividend-adjusted close — the valuation convention, and the only one allowed anywhere
    near the NAV ratio. Network-only; run once to populate the cache.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=False, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.index.name = "date"

        tr = raw[["adj close"]].rename(columns={"adj close": "close"}).dropna()
        px = raw[["close"]].dropna()
        tr.to_parquet(_cache_path(tk, cache_dir, "prices"))
        px.to_parquet(_cache_path(tk, cache_dir, "pxonly"))
        out[tk] = px
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker has BOTH cached parquets (offline-testable)."""
    return all(
        os.path.exists(_cache_path(tk, cache_dir, k))
        for tk in tickers for k in ("prices", "pxonly")
    )


def _load(tickers, cache_dir: str, kind: str, asof: str) -> pd.DataFrame:
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir, kind)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached {kind} for {tk} at {path}. "
                f"Call holdco_nav.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[_safe(tk)] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def load_prices(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                asof: str = AS_OF) -> pd.DataFrame:
    """Cached **total-return** closes, offline. Column names are the safe ticker names."""
    return _load(tickers, cache_dir, "prices", asof)


def load_price_only(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                    asof: str = AS_OF) -> pd.DataFrame:
    """Cached **price-only** (split-adjusted, dividend-unadjusted) closes, offline.

    This is the only series the NAV ratio is allowed to see.
    """
    return _load(tickers, cache_dir, "pxonly", asof)


def cash_index_from_irx(irx: pd.Series) -> pd.Series:
    """Build a daily cash-accrual index from the ^IRX 13-week T-bill discount rate.

    PROXY: ^IRX is a *quoted rate*, not a tradable fund, so this index ignores the small
    slippage a real T-bill ladder suffers. It exists only to extend the cash leg back to
    2004 (BIL starts 2007-05); ``strategy.cash_cross_check`` compares it with BIL's actual
    total return over the overlap.
    """
    r = pd.Series(irx).astype(float).dropna() / 100.0 / TRADING_DAYS_PER_YEAR
    return (1.0 + r).cumprod().rename("cash")


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 16,
    kappa_ann: float = 2.5,          # OU mean-reversion speed of the logit-discount, per year
    disc_mean: float = 0.15,         # long-run discount level
    disc_vol_ann: float = 0.14,      # annualised vol of the discount innovation
    stake_drift: float = 0.08,
    stake_vol: float = 0.25,
    signal_strength: float = 1.0,    # 0 = random-walk discount (the null), 1 = full OU
    start: str = "2010-01-04",
    seed: int = 957,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """One synthetic (stake, holdco) pair with a controllable mean-reverting discount.

    The stake price is a GBM. The discount lives in **logit space**, ``d = 1/(1+exp(-x))``,
    where ``x`` follows a discretised Ornstein-Uhlenbeck process pulled toward
    ``logit(disc_mean)`` at speed ``signal_strength * kappa_ann``. The holdco price is
    ``NAV * (1 - d_t)`` with ``NAV = k * stake``.

    Working in logit space matters for the null: at ``signal_strength=0`` the pull is
    exactly zero, so ``x`` is a *pure driftless random walk* and ``d`` stays inside (0, 1)
    with no reflecting barrier — a barrier would have smuggled a little mean reversion into
    the very control that is supposed to have none.

    Returns ``(frame, truth)`` with columns ``stake``, ``holdco``, ``nav``, ``discount``,
    ``cash``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n)   # OOB-safe: n stays in the low thousands

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    kappa = float(signal_strength) * kappa_ann
    # logit-space vol calibrated so d wanders by roughly disc_vol_ann per year near the mean
    slope = disc_mean * (1.0 - disc_mean)
    sig_x = (disc_vol_ann / slope) * np.sqrt(dt)

    x0 = np.log(disc_mean / (1.0 - disc_mean))
    x = np.empty(n)
    x[0] = x0
    shocks = rng.normal(0.0, sig_x, n)
    for i in range(1, n):
        d_prev = 1.0 / (1.0 + np.exp(-x[i - 1]))
        # Ito correction: the logistic map is convex below its inflection, so a driftless
        # random walk in x would make E[d] creep upward and hand the null a free widening
        # trend. This term cancels it, so at signal_strength=0 the DISCOUNT ITSELF is a
        # martingale and a buy-the-wide-discount rule must earn exactly nothing.
        drift = -0.5 * (sig_x ** 2) * (1.0 - 2.0 * d_prev)
        x[i] = x[i - 1] + kappa * (x0 - x[i - 1]) * dt + drift + shocks[i]
    d = 1.0 / (1.0 + np.exp(-x))

    log_r = rng.normal(stake_drift * dt - 0.5 * stake_vol**2 * dt,
                       stake_vol * np.sqrt(dt), n)
    stake = 100.0 * np.exp(np.cumsum(log_r))
    k = 1.0
    nav = k * stake
    holdco = nav * (1.0 - d)
    cash = np.cumprod(np.full(n, (1.0 + cash_rate_ann) ** dt))

    frame = pd.DataFrame(
        {"stake": stake, "holdco": holdco, "nav": nav, "discount": d, "cash": cash},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": float(signal_strength), "kappa_ann": kappa,
        "disc_mean": disc_mean, "disc_vol_ann": disc_vol_ann, "k": k,
        "n_days": n, "seed": seed, "cash_rate_ann": cash_rate_ann,
        "realised_disc_mean": float(d.mean()), "realised_disc_sd": float(d.std(ddof=1)),
    }
    return frame, truth


def synthetic_panel(
    n_names: int = 7,
    signal_strength: float = 1.0,
    seed: int = 957,
    n_years: int = 16,
    **kw,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A panel of ``n_names`` independent synthetic pairs sharing one calendar.

    Mirrors the real panel's shape (seven names, one common date index) so the same
    portfolio code runs on both. Deterministic given ``seed``.
    """
    frames: dict[str, pd.DataFrame] = {}
    truths = {}
    for i in range(n_names):
        f, t = synthetic_daily(signal_strength=signal_strength, seed=seed + 101 * i,
                               n_years=n_years, **kw)
        frames[f"SYN{i + 1}"] = f
        truths[f"SYN{i + 1}"] = t
    return frames, {"n_names": n_names, "signal_strength": float(signal_strength),
                    "seed": seed, "per_name": truths}
