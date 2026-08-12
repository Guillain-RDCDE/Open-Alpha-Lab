"""Data layer for Study 878 — Economic Policy Uncertainty (EPU).

The claim under test (Baker, Bloom & Davis 2016, *"Measuring Economic Policy
Uncertainty"*, QJE): a newspaper-based index of **policy uncertainty** should carry
information about the future — high EPU should precede **higher equity volatility** and,
per the risk-premium story, **higher forward returns** as compensation for bearing it.

Three ingredients, all offline-friendly once the caches exist.

* **The EPU series (the signal).** The intended source is the free Baker-Bloom-Davis US
  EPU index — monthly ``policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv`` or
  FRED ``USEPUINDXM`` / daily ``USEPUINDXD``. ``fetch_epu()`` tries each in turn (4 retries,
  a real User-Agent) and caches ``_cache/epu_monthly.csv`` on success.

  **Data-honesty note.** From the environment this study was built in, only Yahoo Finance
  was network-reachable (``policyuncertainty.com`` / ``fred.stlouisfed.org`` did not
  resolve). Rather than **fabricate** a newspaper series, we fall back — following the same
  transparent-proxy discipline as [387-economic-surprise-index](../../387-economic-surprise-index/),
  which proxies the proprietary Citi CESI with public data — to a **market-based
  uncertainty proxy built from real VIX** (CBOE implied volatility, a fetchable Yahoo
  series). VIX and EPU co-move (the literature puts the correlation ~0.4-0.6) but VIX is
  **market-implied**, not **text-based**: it is a *labelled proxy*, never presented as the
  newspaper index. ``load_uncertainty()`` returns ``(series, source)`` where ``source`` is
  ``"epu"`` if the real cache is present and ``"vix_proxy"`` otherwise, so every published
  number carries its provenance.

* **The equity tape.** SPY daily adjusted closes (``_cache/spy.csv``) and ^VIX daily closes
  (``_cache/vix.csv``), both via yfinance. Month-end SPY closes drive forward returns;
  daily SPY returns drive monthly realized volatility.

* **Synthetic world — the positive control.** A deterministic, seeded generator
  (:func:`synthetic`) producing a monthly uncertainty index and a daily SPY-like price with
  TUNABLE planted forward edges: ``edge_ret`` injects a genuine uncertainty->forward-return
  relation, ``edge_vol`` a genuine uncertainty->forward-vol relation. ``edge_ret =
  edge_vol = 0`` is the null (uncertainty predicts nothing); the inference must NOT
  manufacture significance there and MUST light up when the edges are planted.

Pure numpy + pandas + stdlib for the offline path. The ``fetch_*`` functions (network) run
once to build the caches and are never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
EPU_CACHE = os.path.join(CACHE_DIR, "epu_monthly.csv")
SPY_CACHE = os.path.join(CACHE_DIR, "spy.csv")
VIX_CACHE = os.path.join(CACHE_DIR, "vix.csv")

START = "1990-01-01"        # VIX begins 1990; SPY 1993
AS_OF = "2026-06-30"        # last complete calendar month at publication

# The real Baker-Bloom-Davis EPU endpoints, tried in order by fetch_epu().
EPU_URLS = [
    ("policyuncertainty",
     "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"),
    ("fred_monthly",
     "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USEPUINDXM"),
    ("fred_daily",
     "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USEPUINDXD"),
]

__all__ = [
    "START", "AS_OF", "CACHE_DIR", "EPU_CACHE", "SPY_CACHE", "VIX_CACHE",
    "fetch", "fetch_epu", "fetch_market",
    "have_real_epu", "have_real_market",
    "load_epu", "load_spy", "load_vix", "load_uncertainty",
    "realized_vol_monthly", "spy_month_end", "build_real", "synthetic",
]


# --------------------------------------------------------------------------- #
# Real tape — fetchers (network; used once to build the caches)
# --------------------------------------------------------------------------- #
def _get(url: str, retries: int = 4, timeout: int = 40) -> str | None:
    """GET a URL text with a real User-Agent and up to ``retries`` attempts."""
    import urllib.request

    for k in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Open-Alpha-Lab research)"}
            )
            return urllib.request.urlopen(req, timeout=timeout).read().decode(
                "utf-8", "replace"
            )
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def _parse_epu_csv(text: str, source: str) -> pd.Series | None:
    """Parse an EPU CSV (policyuncertainty long form or a FRED two-column form) into a
    monthly Series (index=month-end date, name='epu')."""
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    if "year" in cols and "month" in cols:
        # policyuncertainty long form: Year, Month, <index columns...>
        val_cols = [c for c in df.columns
                    if c not in (cols["year"], cols["month"])
                    and pd.api.types.is_numeric_dtype(df[c])]
        if not val_cols:
            return None
        y = pd.to_numeric(df[cols["year"]], errors="coerce")
        m = pd.to_numeric(df[cols["month"]], errors="coerce")
        val = pd.to_numeric(df[val_cols[0]], errors="coerce")
        ok = y.notna() & m.notna()
        idx = pd.to_datetime(
            dict(year=y[ok].astype(int), month=m[ok].astype(int), day=1)
        ) + pd.offsets.MonthEnd(0)
        s = pd.Series(val[ok].to_numpy(), index=idx, name="epu").dropna()
    else:
        # FRED two-column form: DATE, <ID>
        df.columns = ["date", "val"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        s = df.dropna().set_index("date")["val"]
        s = s.resample("ME").mean().dropna()
        s.name = "epu"
    return s.sort_index() if len(s) else None


def fetch_epu(path: str = EPU_CACHE) -> pd.Series | None:
    """Try the real Baker-Bloom-Davis EPU feeds in order; cache a monthly CSV on success.

    Returns the monthly Series, or ``None`` if every endpoint was unreachable (in which case
    NO file is written and no data is fabricated — :func:`load_uncertainty` then falls back
    to the documented VIX proxy)."""
    for source, url in EPU_URLS:
        text = _get(url)
        if text is None:
            continue
        s = _parse_epu_csv(text, source)
        if s is not None and len(s) > 24:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            s.to_csv(path)
            print(f"[fetch_epu] cached {len(s)} months from {source} -> {path}")
            return s
    print("[fetch_epu] all EPU endpoints unreachable — no real newspaper series fetched; "
          "load_uncertainty() will use the documented VIX proxy (labelled 'vix_proxy').")
    return None


def _fetch_yahoo(ticker: str, path: str) -> pd.Series:
    import yfinance as yf

    d = yf.download(ticker, start=START, end="2026-07-01",
                    auto_adjust=True, progress=False)["Close"]
    s = d.iloc[:, 0] if hasattr(d, "columns") else d
    s.name = "val"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.to_csv(path)
    return s


def fetch_market() -> None:
    """Download SPY + ^VIX daily closes and cache them (network; once)."""
    _fetch_yahoo("SPY", SPY_CACHE)
    _fetch_yahoo("^VIX", VIX_CACHE)


def fetch() -> None:
    """Build every real cache: the EPU series (best-effort) and the SPY/VIX tape."""
    fetch_epu()
    fetch_market()


# --------------------------------------------------------------------------- #
# Real tape — offline loaders
# --------------------------------------------------------------------------- #
def have_real_epu(path: str = EPU_CACHE) -> bool:
    return os.path.exists(path)


def have_real_market(spy: str = SPY_CACHE, vix: str = VIX_CACHE) -> bool:
    return os.path.exists(spy) and os.path.exists(vix)


def _load_csv_series(path: str, name: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df.iloc[:, 0].astype(float)
    s.name = name
    return s.dropna()


def load_epu(path: str = EPU_CACHE) -> pd.Series:
    """Cached real EPU monthly series (raises if the real cache is absent)."""
    s = _load_csv_series(path, "epu")
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s


def load_spy(path: str = SPY_CACHE) -> pd.Series:
    return _load_csv_series(path, "spy")


def load_vix(path: str = VIX_CACHE) -> pd.Series:
    return _load_csv_series(path, "vix")


def load_uncertainty(asof: str = AS_OF) -> tuple[pd.Series, str]:
    """The monthly uncertainty signal and its provenance.

    Returns ``(series, source)``. ``source == "epu"`` when the real Baker-Bloom-Davis cache
    is present; otherwise ``source == "vix_proxy"`` and the series is **month-end VIX** — a
    documented market-based proxy (never the newspaper index). Sliced to ``[START, asof]``,
    month-end stamped, named ``unc``.
    """
    if have_real_epu():
        s = load_epu()
        source = "epu"
    else:
        vix = load_vix()
        s = vix.resample("ME").last().dropna()
        source = "vix_proxy"
    s = s[s.index <= pd.Timestamp(asof)]
    s.name = "unc"
    return s, source


def spy_month_end(spy: pd.Series | None = None) -> pd.Series:
    """SPY resampled to month-end closes."""
    if spy is None:
        spy = load_spy()
    me = spy.resample("ME").last().dropna()
    me.name = "spy"
    return me


def realized_vol_monthly(spy: pd.Series | None = None, ann: int = 252) -> pd.Series:
    """Annualised **realized volatility per calendar month** from daily SPY log returns.

    For each month, the sample std of that month's daily log returns times ``sqrt(ann)``.
    Indexed by month-end, named ``rv``. This is the *contemporaneous* monthly vol; the
    strategy layer builds the FORWARD realized vol from it.
    """
    if spy is None:
        spy = load_spy()
    lr = np.log(spy / spy.shift(1)).dropna()
    rv = lr.groupby(lr.index.to_period("M")).std(ddof=0) * np.sqrt(ann)
    rv.index = rv.index.to_timestamp(how="end").normalize()
    rv.index = rv.index + pd.offsets.MonthEnd(0)
    rv.name = "rv"
    return rv.dropna()


def build_real(asof: str = AS_OF) -> tuple[pd.DataFrame, str]:
    """Aligned monthly frame ``[unc, spy, rv]`` from the real caches, plus the source label.

    ``unc`` is the uncertainty signal (real EPU or the VIX proxy), ``spy`` the month-end
    close, ``rv`` the month's annualised realized vol. All stamped at month-end; the
    strategy layer applies the forward shift, so this frame carries no look-ahead by itself.
    """
    unc, source = load_uncertainty(asof)
    spy_daily = load_spy()
    spy_daily = spy_daily[spy_daily.index <= pd.Timestamp(asof)]
    spy = spy_month_end(spy_daily)
    rv = realized_vol_monthly(spy_daily)
    frame = pd.DataFrame({"unc": unc}).join(
        pd.DataFrame({"spy": spy}), how="inner").join(
        pd.DataFrame({"rv": rv}), how="inner")
    return frame.dropna(), source


# --------------------------------------------------------------------------- #
# Synthetic positive control — planted forward edges (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic(
    n_months: int = 360,
    edge_ret: float = 0.0,
    edge_vol: float = 0.0,
    seed: int = 878,
    start: str = "1995-01-02",
    phi: float = 0.75,
    base_daily_vol: float = 0.0085,
    drift: float = 0.05 / 252,
) -> tuple[pd.Series, pd.Series]:
    """Deterministic monthly uncertainty + daily SPY-like price with planted forward edges.

    The uncertainty index ``u`` is a monthly AR(1) (persistence ``phi``) standardised to
    ~unit variance — the mild autocorrelation a real uncertainty index carries. Daily SPY
    returns in month ``m`` have vol ``base_daily_vol * (1 + edge_vol * u[m-1])`` and drift
    ``drift + edge_ret * u[m-1] / 21``: so the *previous* month's uncertainty drives *this*
    month's vol and return — a genuine FORWARD relation when the edges are non-zero.

    ``edge_ret = edge_vol = 0`` is the null: ``u`` is unconditionally-persistent noise that
    predicts neither forward return nor forward vol. Returns ``(spy_daily, unc_monthly)``:
    a daily business-day price Series and a month-end-stamped uncertainty Series. The daily
    index uses ``bdate_range`` well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    n_days = n_months * 21
    idx = pd.bdate_range(start, periods=n_days)
    months = idx.to_period("M")
    uniq = months.unique()
    n_m = len(uniq)

    u = np.empty(n_m)
    u[0] = rng.normal()
    innov_sd = np.sqrt(1.0 - phi ** 2)
    for t in range(1, n_m):
        u[t] = phi * u[t - 1] + rng.normal(0.0, innov_sd)

    # per-day: use the PREVIOUS month's uncertainty (forward relation, no look-ahead)
    month_pos = np.searchsorted(uniq.astype("int64"), months.astype("int64"))
    u_prev_day = np.where(month_pos > 0, u[np.clip(month_pos - 1, 0, n_m - 1)], 0.0)

    vol = base_daily_vol * (1.0 + edge_vol * u_prev_day)
    vol = np.clip(vol, base_daily_vol * 0.2, None)
    z = rng.normal(0.0, 1.0, n_days)
    r = drift + edge_ret * u_prev_day / 21.0 + vol * z
    price = 100.0 * np.cumprod(1.0 + r)

    spy_daily = pd.Series(price, index=idx, name="spy")
    unc_idx = uniq.to_timestamp(how="end").normalize() + pd.offsets.MonthEnd(0)
    unc_monthly = pd.Series(u, index=unc_idx, name="unc")
    return spy_daily, unc_monthly
