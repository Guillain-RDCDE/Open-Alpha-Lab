"""Data layer for Study 956 — the ADR custody (depositary) fee drag.

An American Depositary Receipt is a claim on foreign shares held by a depositary bank.
The bank charges the *holder* a pass-through **custody / depositary service fee** — the
published schedules sit around **1-5 cents per ADS per year** — and nets it out of the
dividend before it reaches the account. On top of that the foreign tax authority
withholds tax on the same dividend. Neither leak appears on a price chart. Both appear
in the **total-return** tape, and only there.

Three tapes, one shape (a date-indexed daily close frame):

- ``fetch`` / ``load_prices`` — daily closes from Yahoo! Finance (``yfinance``) for the
  fifteen ADRs in ``PAIRS``, their home-market ordinary lines, and the six FX crosses that
  translate the home line into dollars (ten of the fifteen pairs survive the coverage
  screen; see ``strategy.coverage_screen``). Two flavours are cached per equity leg:

    * ``prices_<T>_1d.parquet`` — **total-return** close (``auto_adjust=True``:
      split- *and* dividend-adjusted);
    * ``praw_<T>_1d.parquet``  — **price-only** close (``auto_adjust=False``, Yahoo's
      split-adjusted ``Close``).

  The gap between the two is the realised distribution yield, which is what makes the
  withholding-versus-custody split measurable rather than assumed. ``fetch`` touches the
  network (retry up to 4x) and writes into the **shared** desk cache
  ``studies/_cache``; ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic only).

- ``synthetic_pair`` / ``synthetic_panel`` — *deterministic, offline* generators. A home
  line, an FX cross and an ADR whose log price ratio to the FX-adjusted home line is a
  stationary arbitrage band **minus a planted linear drag**. The ``signal_strength`` knob
  scales the planted drag: at ``signal_strength=0`` there is no fee at all (the null — the
  estimator must stay quiet); at ``signal_strength=1`` the planted fee is the full
  ``drag_bps_per_year``. A ``ratio_break`` switch plants an ADS-ratio change so the break
  detector can be tested on a world where we know the truth.

No look-ahead can arise here — the estimand is a *drift*, not a forecast — but the
one-execution-lag discipline still applies to the only traded leg in the study (the
"buy the home line instead" switch in ``strategy.switch_race``), where the signal formed
through day ``t`` is acted on at ``t+1``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a study-local one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# The ten pairs
# --------------------------------------------------------------------------- #
# Each row is (ADR, home line, FX ticker, FX orientation, currency, country,
# assumed treaty withholding rate on dividends paid to a US holder).
#
#   fx_invert=False -> the FX series is USD per unit of local currency (GBPUSD=X)
#   fx_invert=True  -> the FX series is local currency per USD (JPY=X, DKK=X, TWD=X)
#
# WITHHOLDING RATES ARE AN ASSUMPTION / PROXY, not tape. They are the headline treaty
# rates a US holder of the ADR would face; the realised rate a depositary applies can
# differ (relief-at-source versus reclaim, ADS-level pooling, and for India a regime that
# changed in 2020). Every rate is swept in ``strategy.withholding_sweep``. The UK rate is
# zero because the UK levies no dividend withholding tax at all.
PAIRS = (
    dict(adr="TTE", local="TTE.PA", fx="EURUSD=X", fx_invert=False,
         ccy="EUR", country="France", wht=0.15, name="TotalEnergies"),
    dict(adr="SNY", local="SAN.PA", fx="EURUSD=X", fx_invert=False,
         ccy="EUR", country="France", wht=0.15, name="Sanofi"),
    dict(adr="SAP", local="SAP.DE", fx="EURUSD=X", fx_invert=False,
         ccy="EUR", country="Germany", wht=0.15, name="SAP"),
    dict(adr="PHG", local="PHIA.AS", fx="EURUSD=X", fx_invert=False,
         ccy="EUR", country="Netherlands", wht=0.15, name="Philips"),
    dict(adr="ING", local="INGA.AS", fx="EURUSD=X", fx_invert=False,
         ccy="EUR", country="Netherlands", wht=0.15, name="ING Groep"),
    dict(adr="E", local="ENI.MI", fx="EURUSD=X", fx_invert=False,
         ccy="EUR", country="Italy", wht=0.15, name="Eni"),
    dict(adr="NVS", local="NOVN.SW", fx="CHF=X", fx_invert=True,
         ccy="CHF", country="Switzerland", wht=0.15, name="Novartis"),
    dict(adr="NVO", local="NOVO-B.CO", fx="DKK=X", fx_invert=True,
         ccy="DKK", country="Denmark", wht=0.15, name="Novo Nordisk"),
    dict(adr="TM", local="7203.T", fx="JPY=X", fx_invert=True,
         ccy="JPY", country="Japan", wht=0.15315, name="Toyota"),
    dict(adr="TSM", local="2330.TW", fx="TWD=X", fx_invert=True,
         ccy="TWD", country="Taiwan", wht=0.21, name="TSMC"),
    # The five UK pairs below are carried through the *same* pipeline on purpose. They
    # are the study's cautionary exhibit: Yahoo's LSE "adjusted close" is split-adjusted
    # only, so the home leg is missing its entire dividend stream and the naive
    # comparison manufactures a 5 %/yr "custody fee". The coverage screen in
    # ``strategy.coverage_screen`` throws them out automatically — nothing is hand-picked.
    dict(adr="SHEL", local="SHEL.L", fx="GBPUSD=X", fx_invert=False,
         ccy="GBP", country="United Kingdom", wht=0.00, name="Shell"),
    dict(adr="BP", local="BP.L", fx="GBPUSD=X", fx_invert=False,
         ccy="GBP", country="United Kingdom", wht=0.00, name="BP"),
    dict(adr="HSBC", local="HSBA.L", fx="GBPUSD=X", fx_invert=False,
         ccy="GBP", country="United Kingdom", wht=0.00, name="HSBC"),
    dict(adr="UL", local="ULVR.L", fx="GBPUSD=X", fx_invert=False,
         ccy="GBP", country="United Kingdom", wht=0.00, name="Unilever"),
    dict(adr="RIO", local="RIO.L", fx="GBPUSD=X", fx_invert=False,
         ccy="GBP", country="United Kingdom", wht=0.00, name="Rio Tinto"),
)

ADR_TICKERS = tuple(p["adr"] for p in PAIRS)
LOCAL_TICKERS = tuple(dict.fromkeys(p["local"] for p in PAIRS))
FX_TICKERS = tuple(dict.fromkeys(p["fx"] for p in PAIRS))
EQUITY_TICKERS = ADR_TICKERS + LOCAL_TICKERS

# Cash leg for the excess-of-cash race in the tradability section (total return only).
CASH_TICKER = "BIL"


def pair_by_adr(adr: str) -> dict:
    """Look up a pair record by its ADR ticker."""
    for p in PAIRS:
        if p["adr"] == adr:
            return p
    raise KeyError(adr)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance, cache-only by default
# --------------------------------------------------------------------------- #
def _safe(ticker: str) -> str:
    """Cache-safe filename stem, extending the desk convention to foreign tickers.

    The desk convention strips ``^`` and ``=``; home-market tickers additionally carry a
    suffix dot (``SHEL.L``, ``7203.T``), which becomes a dash so the stem stays a plain
    filename token: ``SHEL.L`` -> ``SHEL-L``, ``NOVO-B.CO`` -> ``NOVO-B-CO``.
    """
    return (ticker.replace("=", "").replace("^", "").replace("/", "").replace(".", "-"))


def _cache_path(ticker: str, cache_dir: str, kind: str = "tr") -> str:
    prefix = "prices" if kind == "tr" else "praw"
    return os.path.join(cache_dir, f"{prefix}_{_safe(ticker)}_1d.parquet")


def _download(tk: str, start: str, end, auto_adjust: bool, retries: int) -> pd.DataFrame:
    import yfinance as yf  # lazy: only when we actually go to the network

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(tk, start=start, end=end, interval="1d",
                              auto_adjust=auto_adjust, progress=False)
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no data for {tk}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    df = raw[["close"]].copy()
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df.index = df.index.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["close"])


def fetch(
    start: str = "2000-01-01",
    end: str | None = "2026-07-01",
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict:
    """Download and cache every leg this study needs. Network-only; run once.

    For each equity leg (ADR and home line) two series are cached: the **total-return**
    close (``auto_adjust=True``) and the **price-only** close (``auto_adjust=False``).
    FX crosses are cached total-return-flavoured only (a spot cross has no dividend).
    """
    os.makedirs(cache_dir, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for tk in EQUITY_TICKERS:
        for kind, auto in (("tr", True), ("px", False)):
            df = _download(tk, start, end, auto, retries)
            df.to_parquet(_cache_path(tk, cache_dir, kind))
            out[f"{tk}:{kind}"] = df
    for tk in FX_TICKERS + (CASH_TICKER,):
        df = _download(tk, start, end, True, retries)
        df.to_parquet(_cache_path(tk, cache_dir, "tr"))
        out[f"{tk}:tr"] = df
    return out


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every leg this study needs is already on disk (offline-testable)."""
    ok = all(os.path.exists(_cache_path(tk, cache_dir, k))
             for tk in EQUITY_TICKERS for k in ("tr", "px"))
    return ok and all(os.path.exists(_cache_path(tk, cache_dir, "tr"))
                      for tk in FX_TICKERS + (CASH_TICKER,))


def load_prices(
    tickers,
    cache_dir: str = DEFAULT_CACHE,
    kind: str = "tr",
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily closes OFFLINE into one aligned frame (one column per ticker).

    ``kind="tr"`` returns **total-return** closes, ``kind="px"`` **price-only** closes.
    Sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError`` on a
    cache miss — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir, kind)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached {kind} series for {tk} at {path}. "
                f"Call adr_drag.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s[~s.index.duplicated(keep="last")]
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def clean_mask(x: pd.Series, window: int = 11, thresh: float = 0.15) -> pd.Series:
    """Flag bad prints: log-ratio observations far from their own local median.

    Yahoo's foreign series carry occasional corrupt closes — ``TWD=X`` prints 1.80 and
    3.67 against a 30-ish level, ``NOVO-B.CO`` has a scatter of 100 %+ one-day round
    trips in the early 2000s. Against the ADR's price-only ratio, which is a *tight*
    arbitrage band, any observation more than ``thresh`` log points from its own centred
    rolling median is such a print, not a market move.

    The median window is **centred**, so this step peeks at neighbouring days. LOOK-AHEAD,
    stated plainly, and its consequences measured rather than waved away:

    * On the **headline estimand** it is inert. The pooled income gap is +13.80 bp/yr with
      the filter on and +13.80 bp/yr with it off (``clean_thresh=None``): a trend fit on a
      *level* cannot be moved by 24 isolated bad prints out of 83,728 rows.
    * On the **one traded leg** (:func:`strategy.switch_race`) it is decisive, and in the
      direction that makes the trade look *worse*: those same 24 rows (0.03 % of the
      panel) swing the home-versus-ADR race from +944.7 bp/yr to +14.6 bp/yr, because an
      arithmetic mean of daily returns is dominated by a corrupt round trip. So that leg
      is a **measurement, not an implementable backtest** — it is reported as Mirage, and
      it would be Mirage either way (raw HAC t = +1.05).

    A genuine ADS-ratio *step* survives the filter (the median follows within a few days),
    and the break detector handles the step.
    """
    med = x.rolling(window, center=True, min_periods=3).median()
    return (x - med).abs() > thresh


def load_pair(pair: dict, cache_dir: str = DEFAULT_CACHE, asof: str = AS_OF,
              clean_thresh: float = 0.15) -> pd.DataFrame:
    """Assemble one ADR / home-line / FX triple onto a common calendar.

    Returns a frame with columns ``adr_tr``, ``adr_px`` (the ADR's total-return and
    price-only closes, USD), ``loc_tr``, ``loc_px`` (the home line, **local currency**),
    ``fx`` (USD per unit of local currency, already oriented), and the derived
    ``loc_tr_usd`` / ``loc_px_usd``. The index is the **intersection** of the three
    calendars — no forward-filling across a closed exchange, which would otherwise plant
    a spurious step in the ratio around every foreign holiday. Rows flagged by
    :func:`clean_mask` are dropped from every column together; ``clean_thresh=None``
    disables the filter (it is swept in ``strategy``-side robustness).
    """
    adr_tr = load_prices([pair["adr"]], cache_dir, "tr", asof)[pair["adr"]]
    adr_px = load_prices([pair["adr"]], cache_dir, "px", asof)[pair["adr"]]
    loc_tr = load_prices([pair["local"]], cache_dir, "tr", asof)[pair["local"]]
    loc_px = load_prices([pair["local"]], cache_dir, "px", asof)[pair["local"]]
    fx = load_prices([pair["fx"]], cache_dir, "tr", asof)[pair["fx"]]
    if pair["fx_invert"]:
        fx = 1.0 / fx

    idx = adr_tr.dropna().index
    for s in (adr_px, loc_tr, loc_px, fx):
        idx = idx.intersection(s.dropna().index)
    idx = idx.sort_values()

    df = pd.DataFrame({
        "adr_tr": adr_tr.reindex(idx),
        "adr_px": adr_px.reindex(idx),
        "loc_tr": loc_tr.reindex(idx),
        "loc_px": loc_px.reindex(idx),
        "fx": fx.reindex(idx),
    })
    df["loc_tr_usd"] = df["loc_tr"] * df["fx"]
    df["loc_px_usd"] = df["loc_px"] * df["fx"]
    df.index.name = "date"
    df = df.dropna()
    if clean_thresh is not None and len(df) > 20:
        x_px = np.log(df["adr_px"].astype(float)) - np.log(df["loc_px_usd"].astype(float))
        df = df[~clean_mask(x_px, thresh=clean_thresh).to_numpy()]
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(df.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — deterministic offline core
# --------------------------------------------------------------------------- #
_BDATE_CACHE: dict = {}


def _bdays(start: str, periods: int) -> pd.DatetimeIndex:
    """Memoised business-day index. ``pd.bdate_range`` costs ~0.8 s for 15 years of bars
    and every synthetic panel asks for the same calendar, so it is built once per key.
    ``periods`` stays well under 10,000, which keeps the index inside pandas' ns range on
    pandas 2.x / Python 3.10 (a large ``periods=`` there overflows).
    """
    key = (start, int(periods))
    if key not in _BDATE_CACHE:
        _BDATE_CACHE[key] = pd.bdate_range(start=start, periods=int(periods))
    return _BDATE_CACHE[key]


def synthetic_pair(
    n_years: int = 15,
    drag_bps_per_year: float = 25.0,   # planted total drag, bps of NAV per year
    gross_yield: float = 0.035,        # home-line gross dividend yield
    wht: float = 0.15,                 # planted withholding rate on that dividend
    band_sd: float = 0.010,            # sd of the stationary ADR/local arbitrage band
    band_phi: float = 0.90,            # daily AR(1) persistence of the band
    equity_vol: float = 0.25,
    equity_drift: float = 0.07,
    fx_vol: float = 0.09,
    fx_drift: float = 0.00,
    signal_strength: float = 1.0,      # 0 = no fee at all (the null), 1 = full planted drag
    ratio_break: float = 0.0,          # log-size of a planted ADS-ratio change at mid-sample
    pays_per_year: int = 2,            # dividend frequency
    adr_pay_lag: int = 25,             # business days between the home ex-date and the ADS credit
    pay_jitter: float = 0.05,          # relative noise on each converted ADS payment
    start: str = "2006-01-02",
    seed: int = 956,
) -> tuple[pd.DataFrame, dict]:
    """One ADR / home-line / FX triple with a **known** custody-fee drag planted in it.

    Construction, in the same order the real tape produces it:

    1. a home-line price path and an FX path (both GBM);
    2. the home line pays a **gross** dividend ``pays_per_year`` times a year; its
       total-return leg steps on each ex-date;
    3. the ADR receives the *same* economic dividend ``adr_pay_lag`` business days later,
       less the planted **withholding** and the planted **custody fee**, and with a small
       multiplicative jitter standing in for the depositary's FX conversion — so the two
       income legs step on *different* days and by *slightly* different amounts, which is
       the whole measurement difficulty on the real tape;
    4. a stationary AR(1) arbitrage band (the ADR does not close in lockstep with a market
       that shut hours earlier) is layered on **both** ADR legs, so it cancels out of a
       total-return-minus-price-only difference exactly as it does on the tape;
    5. optionally a one-off ADS-ratio jump of ``ratio_break`` log points at mid-sample.

    The fee and the tax are netted from the *dividend*, never from the share price, so the
    price ratio stays a clean placebo — which is precisely the property the real-tape
    estimator leans on.

    ``signal_strength`` scales *both* the custody fee and the withholding leak, so at
    ``signal_strength=0`` the ADR is a perfect, fee-free claim on the home line and the
    estimator must report a drag indistinguishable from zero.

    Returns ``(frame, truth)`` with the same columns as :func:`load_pair` plus a ``truth``
    dict recording the planted drag, split into its custody and withholding halves.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = _bdays(start, n)  # OOB-safe: n stays well under 10k

    ss = float(signal_strength)
    custody = ss * drag_bps_per_year * 1e-4          # per year, fraction of NAV
    wht_eff = ss * wht
    wht_drag = wht_eff * gross_yield                 # per year, fraction of NAV
    total_drag = custody + wht_drag

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    ev, fv = equity_vol * np.sqrt(dt), fx_vol * np.sqrt(dt)
    loc_px = 100.0 * np.exp(np.cumsum(
        rng.normal(equity_drift * dt - 0.5 * ev * ev, ev, n)))
    fx = 1.5 * np.exp(np.cumsum(rng.normal(fx_drift * dt - 0.5 * fv * fv, fv, n)))

    # Discrete dividend legs. q = the home line's per-payment gross yield; q_adr = what
    # reaches the ADS holder after withholding and the custody fee.
    step = max(int(TRADING_DAYS_PER_YEAR // pays_per_year), 1)
    ex_days = np.arange(step, n, step)
    q = gross_yield / pays_per_year
    q_adr_base = (gross_yield * (1.0 - wht_eff) - custody) / pays_per_year

    loc_income = np.zeros(n)
    adr_income = np.zeros(n)
    for k, d in enumerate(ex_days):
        loc_income[d:] += np.log1p(q)
        jit = 1.0 + pay_jitter * rng.standard_normal()
        pay = d + adr_pay_lag
        if pay < n:
            adr_income[pay:] += np.log1p(max(q_adr_base * jit, -0.9))

    net_yield = gross_yield * (1.0 - wht_eff)
    loc_tr = loc_px * np.exp(loc_income)

    # Stationary arbitrage band on the ADR (AR(1)), started from its stationary
    # distribution so the path carries no systematic drift from a zero initial condition.
    band = np.zeros(n)
    band[0] = rng.normal(0.0, band_sd)
    innov = rng.normal(0.0, band_sd * np.sqrt(1.0 - band_phi ** 2), n)
    for i in range(1, n):
        band[i] = band_phi * band[i - 1] + innov[i]

    ratio_step = np.zeros(n)
    if ratio_break != 0.0:
        ratio_step[n // 2:] = ratio_break

    adr_px = loc_px * fx * np.exp(band + ratio_step)
    adr_tr = adr_px * np.exp(adr_income)

    df = pd.DataFrame(
        {"adr_tr": adr_tr, "adr_px": adr_px, "loc_tr": loc_tr, "loc_px": loc_px, "fx": fx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    df["loc_tr_usd"] = df["loc_tr"] * df["fx"]
    df["loc_px_usd"] = df["loc_px"] * df["fx"]

    # The exactly planted income-gap slope, in NAV fraction per year.
    planted_gap = pays_per_year * (np.log1p(q) - np.log1p(q_adr_base))
    truth = {
        "seed": seed, "n_days": n, "n_years": n_years,
        "signal_strength": ss,
        "custody_drag_per_year": custody,
        "wht_drag_per_year": wht_drag,
        "total_drag_per_year": total_drag,
        "planted_gap_per_year": float(planted_gap),
        "gross_yield": gross_yield,
        "net_yield": net_yield,
        "wht": wht_eff,
        "ratio_break": ratio_break,
        "band_sd": band_sd,
        "pays_per_year": pays_per_year,
        "adr_pay_lag": adr_pay_lag,
    }
    return df, truth


def synthetic_panel(
    n_names: int = 10,
    drag_bps_per_year: float = 25.0,
    signal_strength: float = 1.0,
    n_years: int = 15,
    seed: int = 956,
    **kwargs,
) -> tuple[dict, dict]:
    """``n_names`` independent synthetic pairs sharing one planted drag.

    Returns ``(frames, truth)`` where ``frames`` maps a fake ADR ticker (``SYN00`` ...)
    to a :func:`synthetic_pair` frame. Used to test the *pooled* estimator: the planted
    drag is identical across names, so the cross-name mean must recover it and the
    cross-name dispersion must be pure estimation noise.
    """
    frames, truths = {}, {}
    for i in range(n_names):
        df, tr = synthetic_pair(
            n_years=n_years, drag_bps_per_year=drag_bps_per_year,
            signal_strength=signal_strength, seed=seed + 100 * i, **kwargs,
        )
        tag = f"SYN{i:02d}"
        frames[tag] = df
        truths[tag] = tr
    truth = dict(truths[f"SYN00"])
    truth["n_names"] = n_names
    truth["per_name"] = truths
    return frames, truth
