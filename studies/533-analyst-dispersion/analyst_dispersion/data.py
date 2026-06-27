"""Data layer for Study 533 (Analyst-Dispersion / Diether-Malloy-Scherbina puzzle).

Two tapes, one schema. Each carries (a) a per-name **forecast-dispersion** score and (b) a
panel of realised returns to sort against.

- **Real snapshot.** yfinance ``Ticker.earnings_estimate`` gives, per name, the **current**
  consensus EPS estimate for the current fiscal year (``0y``) with its analyst **low / mean /
  high**. We form the Diether-Malloy-Scherbina dispersion proxy

        dispersion = (high − low) / |mean|

  the cross-sectional spread of opinion scaled by the level — the free analogue of the academic
  measure (std-dev of forecasts / |mean forecast|). Daily adjusted-close prices (a long pull)
  give the trailing-return panel. Both are cached under this study's own ``_cache/``.

  **Honest limitation.** yfinance exposes only the *current* estimate snapshot — there is **no
  historical dispersion series**. So the dispersion column is a single cross-section stamped on
  the pull date; we can sort it against *trailing* returns (a contemporaneous DMS association)
  but cannot build a tradable month-by-month *forward* panel the way the academic study does.
  This is named on the Signal axis and is the central caveat of the whole study.

- **Synthetic.** A deterministic, fixed-seed generator that builds a multi-period return panel
  in which dispersion **does** predict low returns by a tunable ``dms_edge`` knob (knob = 0 is
  the null). It is the positive control: the cross-sectional sort engine must recover a planted
  dispersion→return relationship and must NOT manufacture one from noise.

Pure numpy + pandas + stdlib for the offline path. ``fetch_snapshot`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")
DISP_CACHE = os.path.join(DEFAULT_CACHE, "disp_snapshot.csv")
PRICES_CACHE = os.path.join(DEFAULT_CACHE, "disp_prices.parquet")

# A fixed, transparent basket of large, liquid, well-covered US large-caps with deep analyst
# coverage on yfinance — chosen for a wide spread of forecast dispersion (steady consumer
# staples with tight estimates at one end; cyclical/energy/high-growth names with wide
# estimates at the other). Survivors basket (all still trading in 2026) — named on the
# Signal axis.
BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC",
    "XOM", "CVX", "COP", "JNJ", "PFE", "MRK", "UNH", "PG", "KO", "PEP",
    "WMT", "HD", "MCD", "DIS", "NFLX", "CRM", "ORCL", "INTC", "CSCO", "TXN",
    "BA", "CAT", "GE", "MMM", "HON", "GS", "MS", "C", "T", "VZ",
]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    dms_edge: float  # how strongly high dispersion drags future returns DOWN (DMS sign)

    @property
    def has_edge(self) -> bool:
        return self.dms_edge != 0.0


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_snapshot(cache_dir: str = DEFAULT_CACHE, start: str = "2015-01-01",
                   end: str | None = None, retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull per-name analyst EPS-estimate dispersion + daily prices, and cache them.

    Network-only; used once to build the cache. Writes a long dispersion CSV (one row per
    name carrying mean/low/high EPS estimate, analyst count, and the dispersion proxy) and a
    wide adjusted-close parquet. Guards yfinance flakiness with retries.

    The dispersion proxy is ``(high − low) / |mean|`` on the current-fiscal-year (``0y``)
    consensus EPS estimate — the Diether-Malloy-Scherbina spread of opinion.
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    rows = []
    for tk in BASKET:
        ee = None
        for _ in range(retries):
            try:
                ee = yf.Ticker(tk).earnings_estimate
                if ee is not None and len(ee):
                    break
            except Exception:  # noqa: BLE001
                ee = None
        if ee is None or "0y" not in ee.index:
            continue
        r = ee.loc["0y"]
        avg = float(r.get("avg", np.nan))
        lo = float(r.get("low", np.nan))
        hi = float(r.get("high", np.nan))
        n_an = float(r.get("numberOfAnalysts", np.nan))
        if not np.isfinite(avg) or avg == 0 or not np.isfinite(lo) or not np.isfinite(hi):
            continue
        disp = (hi - lo) / abs(avg)
        rows.append({"ticker": tk, "eps_mean": avg, "eps_low": lo, "eps_high": hi,
                     "n_analysts": n_an, "dispersion": float(disp)})
    disp = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    disp.to_csv(os.path.join(cache_dir, "disp_snapshot.csv"), index=False)

    keep = list(disp["ticker"])
    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(keep, start=start, end=end, auto_adjust=True,
                              progress=False, threads=True)["Close"]
            if raw is not None and not raw.empty:
                break
        except Exception:  # noqa: BLE001
            raw = None
    prices = raw.dropna(how="all") if raw is not None else pd.DataFrame()
    if not prices.empty:
        prices.to_parquet(os.path.join(cache_dir, "disp_prices.parquet"))
    return disp, prices


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return (os.path.exists(os.path.join(cache_dir, "disp_snapshot.csv"))
            and os.path.exists(os.path.join(cache_dir, "disp_prices.parquet")))


def load_real(cache_dir: str = DEFAULT_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached dispersion snapshot + wide daily-price frame."""
    disp = pd.read_csv(os.path.join(cache_dir, "disp_snapshot.csv"))
    prices = pd.read_parquet(os.path.join(cache_dir, "disp_prices.parquet")).sort_index()
    # keep only names present in both
    common = [t for t in disp["ticker"] if t in prices.columns]
    disp = disp[disp["ticker"].isin(common)].reset_index(drop=True)
    prices = prices[common]
    return disp, prices


def trailing_returns(prices: pd.DataFrame, disp: pd.DataFrame,
                     horizons=(21, 63, 126, 252)) -> pd.DataFrame:
    """Attach the trailing H-trading-day return realised *into* the snapshot date per name.

    The dispersion column is a single current cross-section; we sort it against the return the
    name realised over the trailing ``H`` days up to the last available close. ``ret_H`` =
    ``close[-1] / close[-1-H] − 1`` per ticker. Returns the dispersion frame with one
    ``ret_<H>`` column per horizon (names with too-short history dropped).
    """
    out = disp.copy()
    for h in horizons:
        col = []
        for tk in out["ticker"]:
            px = prices[tk].dropna().values
            if len(px) <= h:
                col.append(np.nan)
            else:
                col.append(px[-1] / px[-1 - h] - 1.0)
        out[f"ret_{h}"] = col
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 40, n_periods: int = 120, dms_edge: float = 0.0,
                    seed: int = 533, idio_vol: float = 0.010) -> tuple[pd.DataFrame, np.ndarray]:
    """Deterministic multi-period dispersion→return panel with a planted DMS edge.

    Each name draws a fixed **dispersion** score (cross-sectional, in [0, 1]). Over
    ``n_periods`` periods, each name's per-period return is noise *minus* ``dms_edge`` times
    its (de-meaned) dispersion — so with ``dms_edge > 0`` high-dispersion names earn LOWER
    returns (the DMS sign), and with ``dms_edge = 0`` dispersion carries no information.

    Returns ``(panel, dispersion)`` where ``panel`` is a long frame with columns
    ``ticker, period, dispersion, ret`` (the per-period return) and ``dispersion`` is the
    per-name score array — the cross-sectional sort engine consumes the same shape as the real
    trailing-return frame.
    """
    rng = np.random.default_rng(seed)
    names = [f"N{i:02d}" for i in range(n_names)]
    dispersion = rng.uniform(0.05, 0.95, size=n_names)
    disp_dm = dispersion - dispersion.mean()

    rows = []
    # decorative monthly labels via period_range (no ns overflow)
    periods = pd.period_range("2010-01", periods=n_periods, freq="M")
    # The DMS drag is applied per-period and scaled by 1/n_periods so the *cumulative* spread
    # over the panel is on the order of dms_edge (no compounding blow-up): a name's per-period
    # return is base noise minus (dms_edge / n_periods) * de-meaned dispersion.
    drag = dms_edge / n_periods
    for j, name in enumerate(names):
        base = rng.normal(0.004, idio_vol, size=n_periods)      # ~0.4%/mo drift + idio
        rets = base - drag * disp_dm[j]                         # DMS drag (per period)
        for p, r in zip(periods, rets):
            rows.append({"ticker": name, "period": str(p),
                         "dispersion": float(dispersion[j]), "ret": float(r)})
    panel = pd.DataFrame(rows)
    return panel, dispersion


def synthetic_cross_section(dms_edge: float = 0.0, seed: int = 533,
                            n_names: int = 40, n_periods: int = 120) -> pd.DataFrame:
    """One-cross-section view of the synthetic panel: per-name dispersion + cumulative return.

    Collapses :func:`synthetic_panel` to one row per name (``dispersion`` and the compounded
    ``ret_cum`` over all periods) so the same :func:`strategy.long_short` used on the real
    trailing-return frame can be applied directly. Mirrors the real ``ret_<H>`` shape with a
    single ``ret_cum`` column.
    """
    panel, _ = synthetic_panel(n_names=n_names, n_periods=n_periods,
                               dms_edge=dms_edge, seed=seed)
    g = panel.groupby("ticker")
    cum = g["ret"].apply(lambda x: float((1.0 + x).prod() - 1.0))
    disp = g["dispersion"].first()
    out = pd.DataFrame({"ticker": cum.index, "dispersion": disp.values,
                        "ret_cum": cum.values}).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Fingerprint helper
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """Short content fingerprint of a frame (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.select_dtypes("number").fillna(0).to_numpy())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
