"""Data layer for Study 656 — Dragon Portfolio.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily auto-adjusted (total-return) closes for seven tickers, all from
  yfinance (no key), cached as CSV under the study's own ``_cache/``:

  - **SPY** — equity sleeve.
  - **TLT** — long-duration Treasuries, the fixed-income sleeve.
  - **GLD** — physical gold, the crisis/inflation sleeve.
  - **DBC** — a broad commodity index, the *raw material* for the commodity-trend
    sleeve (a 12-month time-series-momentum overlay is applied on top in
    ``strategy.py`` — Chris Cole's Dragon Portfolio wants a diversified multi-market
    systematic trend program across dozens of futures curves; one long/flat overlay on
    a single commodity *index* is a materially weaker proxy, named honestly below).
  - **VXX** — iPath Series B S&P 500 VIX Short-Term Futures ETN, the *crude, decaying*
    stand-in for "long volatility". VXX holds front-month VIX futures and bleeds on
    contango almost every month it isn't paid off by a spike — it is **not** what a
    real long-vol book (OTM SPX puts / variance swaps, actively rolled and sized) would
    hold. **Named data quirk:** the product traces to Jan-2009, but Barclays halted and
    relaunched it as "Series B" in Jan-2018 and yfinance's VXX tape starts **2018-01-25**
    — nearly a decade shorter than the product's own history, and it is the tightest
    constraint on the whole 5-sleeve backtest (see ``FULL_START`` below).
  - **SHY** — 1-3 month/short Treasury ETF, doubling as (a) the cash leg the commodity-
    trend overlay parks in when it's flagged "flat", and (b) the risk-free proxy for
    every excess-of-cash Sharpe.
  - **DBMF** — iMGP DBi Managed Futures Strategy ETF, a *real* multi-strategy managed-
    futures fund (live 2019-05-07). Not a Dragon sleeve — a side-by-side honesty check:
    does our single-index DBC trend overlay behave anything like an actual managed-
    futures program over the period both exist?

* **No synthetic real-tape substitute.** Every sleeve here has deep, liquid, dividend-
  adjusted yfinance history — there is no missing-data case that forces a synthetic
  stand-in for the Signal axis. The synthetic world below is a **machinery / power
  check only** (a deterministic multi-asset generator with a tunable "crisis hedge"
  knob) — never quoted in support of the real-tape verdict.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

TICKERS = ["SPY", "TLT", "GLD", "DBC", "VXX", "SHY", "DBMF"]
CACHE = {t: os.path.join(CACHE_DIR, f"dragon_{t.lower()}.csv") for t in TICKERS}

# Every sleeve's inception (public record) — the honest binding constraints on the
# joint window. VXX is the tightest constraint on the full 5-sleeve Dragon: the product
# dates to 2009-01-30, but yfinance's own VXX tape starts 2018-01-25 (Barclays halted
# and relaunched it as "Series B" in Jan-2018) — the binding constraint used below is
# the DATA constraint (2018), not the product's nominal launch date. DBMF (a side-check
# only, not a sleeve) is tighter still.
INCEPTION = {
    "SPY": "1993-01-29", "TLT": "2002-07-30", "GLD": "2004-11-18",
    "DBC": "2006-02-03", "VXX": "2009-01-30", "SHY": "2002-07-30",
    "DBMF": "2019-05-07",
}
VXX_DATA_START = "2018-01-25"  # first bar yfinance actually serves for VXX (see above)

AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07-10 run)
CORE_START = "2007-02-01"   # first day the 12-month DBC trend signal is fully warmed up
FULL_START = "2018-03-01"   # first full month after yfinance's VXX tape begins (2018-01-25)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2002-01-01", end: str = "2026-07-01") -> None:
    """Download total-return (auto-adjusted) daily closes for all 7 tickers; cache them.

    Network; run once. ``auto_adjust=True`` folds dividends/coupons into the price
    series so pct-change gives a genuine total return per sleeve.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in TICKERS:
        px = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        px[["Close"]].dropna().to_csv(CACHE[t])


def have_real(required: tuple[str, ...] = ("SPY", "TLT", "GLD", "DBC", "VXX", "SHY")) -> bool:
    return all(os.path.exists(CACHE[t]) for t in required)


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: Close series} sliced to [inception, asof]. DBMF included if cached."""
    out: dict[str, pd.Series] = {}
    for t in TICKERS:
        if not os.path.exists(CACHE[t]):
            continue
        s = pd.read_csv(CACHE[t], index_col=0, parse_dates=True).sort_index()["Close"]
        out[t] = s.loc[s.index <= asof]
    return out


def prices_frame(asof: str = AS_OF,
                  tickers: tuple[str, ...] = ("SPY", "TLT", "GLD", "DBC", "VXX", "SHY"),
                  ) -> pd.DataFrame:
    """Outer-joined total-return price frame for ``tickers`` (NaN before each inception)."""
    real = load_real(asof=asof)
    cols = {t: real[t] for t in tickers if t in real}
    return pd.DataFrame(cols).sort_index()


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content hash of a price frame (see quantlab.repro.fingerprint for the
    canonical desk version; this local copy keeps the module import-free of quantlab
    so the study's own data layer has zero cross-package coupling)."""
    import hashlib

    h = hashlib.sha256()
    for c in sorted(frame.columns):
        h.update(str(c).encode())
        h.update(b"\x1f")
        for ts, v in frame[c].items():
            h.update(str(pd.Timestamp(ts).date()).encode())
            token = "nan" if pd.isna(v) else f"{round(float(v), 6):.6f}"
            h.update(token.encode())
            h.update(b",")
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — a faithful-engine / power check ONLY, never real-tape evidence
# --------------------------------------------------------------------------- #
def synthetic_world(hedge_strength: float = 0.0, seed: int = 656,
                     n_months: int = 240, crisis_p: float = 0.04,
                     ) -> tuple[pd.DataFrame, dict]:
    """Deterministic monthly 5-asset world with a TUNABLE planted "crisis hedge".

    Five legs — STK, BOND, GOLD, TREND, VOL — share a base monthly Sharpe/vol close
    to their real analogues. In a random ``crisis_p`` share of months a shared shock
    hits STK hard (a crash month); the two "crisis alpha" legs (TREND, VOL) get an
    extra positive shock of size ``hedge_strength`` **only in crisis months**, while
    BOND/GOLD get a smaller, mixed-sign one (deflationary bonds help, inflationary
    crises hurt gold — kept neutral here to isolate the TREND/VOL hedge channel).

    ``hedge_strength = 0`` is the null: crisis months hit every leg with no offsetting
    diversification, so a Dragon-style blend should show **no** significant crash-
    drawdown reduction vs 60/40. ``hedge_strength > 0`` plants a real crisis-alpha
    effect the harness must recover. Monthly, ~20 years — far below the pandas
    ns-timestamp span trap. Returns (returns frame, truth dict).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2005-01", periods=n_months, freq="M")

    ann_mu = np.array([0.08, 0.03, 0.04, 0.03, -0.30])          # STK BOND GOLD TREND VOL
    ann_vol = np.array([0.16, 0.09, 0.15, 0.10, 0.90])
    mu = ann_mu / 12.0
    vol = ann_vol / np.sqrt(12.0)

    is_crisis = rng.random(n_months) < crisis_p
    z = rng.standard_normal((n_months, 5))
    ret = mu + vol * z

    # shared crisis shock: equities get hit hard; hedge legs get an offsetting bump
    # sized by hedge_strength (0 = no hedge, i.e. the crisis just hits everyone).
    crisis_stk_shock = -0.18
    ret[is_crisis, 0] += crisis_stk_shock
    ret[is_crisis, 3] += hedge_strength * 0.35    # TREND crisis-alpha
    ret[is_crisis, 4] += hedge_strength * 1.10    # VOL crisis-alpha (long-vol spikes hard)
    ret[is_crisis, 1] += 0.02                     # a mild flight-to-quality bond bid, always on

    cols = ["STK", "BOND", "GOLD", "TREND", "VOL"]
    frame = pd.DataFrame(ret, index=idx, columns=cols)
    truth = {"hedge_strength": hedge_strength, "seed": seed, "n_months": n_months,
             "n_crisis": int(is_crisis.sum()), "crisis_p": crisis_p}
    return frame, truth
