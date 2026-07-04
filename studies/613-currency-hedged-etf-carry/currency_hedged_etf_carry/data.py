"""Data layer for Study 613 — Currency-Hedged ETF Carry.

The claim under test: **the currency hedge inside a hedged equity ETF mechanically pockets the
US-vs-foreign short-rate differential** — "free carry hidden in a share class". A hedged ETF
sells the foreign currency forward; covered interest parity prices that forward at (roughly) the
short-rate differential, so

    hedged_return  ~  local_equity_return + (r_US - r_foreign)
    unhedged_return ~ local_equity_return + fx_return
    =>  hedged - unhedged  ~  (r_US - r_foreign) - fx_return          (log-approx)

We test the identity on three live ETF share-class pairs (yfinance, total-return closes):

  * **DXJ vs EWJ** (Japan, 2010-04+) — WisdomTree Japan Hedged Equity vs iShares MSCI Japan. The
    longest tape, but the *baskets differ* (DXJ is dividend-weighted exporter-tilted), which adds
    basket noise on top of the hedge mechanics. DXJ only became a hedged fund on 2010-04-01
    (before that it was the unhedged Japan Total Dividend Fund), so its window starts there.
  * **HEWJ vs EWJ** (Japan, 2014+) — iShares Currency-Hedged MSCI Japan literally *holds EWJ plus
    one-month JPY forwards*: the same basket, so the return differential is the hedge P&L almost
    purely. The clean mechanical pair.
  * **HEDJ vs VGK** (Europe, 2012+) — WisdomTree Europe Hedged Equity vs Vanguard FTSE Europe
    (baskets differ, exporter tilt again; the EUR leg of the story).

Rates:

  * **US short rate** — ^IRX (13-week T-bill discount, %) from yfinance, monthly.
  * **Japan / euro-area short rates** — HARDCODED policy-rate step tables (sources below):
    BOJ uncollateralized overnight call-rate target; ECB deposit facility rate (the binding
    short rate under excess liquidity). Coarse (+/- ~15 bp vs 3M interbank) but the carry story
    lives at the 1-5 %/yr scale, far above that tolerance.

Synthetic world: a deterministic monthly generator with a TUNABLE planted carry (knob
``carry_annual``) plus basket/tracking noise and an fx leg — the positive control (and, at
``carry_annual = 0``, the null that must NOT fire). Index built with ``period_range`` (spans
well under 250 years).

Cache-first: ``fetch_tape`` (network, yfinance) runs once and writes ``_cache/chc_prices.csv``;
everything else is offline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "chc_prices.csv")

# Everything the study needs, one wide frame: ETF total-return closes, spot FX, US 3M bill.
# JPY=X is USDJPY (yen per dollar); EURUSD=X is dollars per euro; ^IRX is the 13-week T-bill
# discount rate in percent.
TICKERS = ["DXJ", "EWJ", "HEWJ", "HEDJ", "VGK", "JPY=X", "EURUSD=X", "^IRX"]

# (hedged, unhedged, currency, label) — the three share-class pairs.
# NOTE mandate changes: DXJ traded from 2006-06 as the UNHEDGED WisdomTree Japan Total Dividend
# Fund and only adopted the currency-hedged mandate on 2010-04-01 (wisdomtree.com; confirmed on
# the tape: hedge beta -0.15 pre-2010-04 vs +0.94 after) — all DXJ windows start 2010-04.
# HEDJ likewise ran a different (unhedged) mandate before Jan-2012 — HEDJ windows start 2012-02.
PAIRS = {
    "DXJ/EWJ": ("DXJ", "EWJ", "JPY", "Japan 2010+ (different baskets; hedged mandate from Apr-2010)"),
    "HEWJ/EWJ": ("HEWJ", "EWJ", "JPY", "Japan 2014+ (SAME basket: HEWJ holds EWJ + forwards)"),
    "HEDJ/VGK": ("HEDJ", "VGK", "EUR", "Europe 2012+ (different baskets)"),
}

AS_OF = "2026-06-30"  # last complete calendar month at build time (2026-07-03)

# --------------------------------------------------------------------------- #
# Hardcoded foreign short-rate step tables (annual %, effective from date)
# --------------------------------------------------------------------------- #
# Bank of Japan — uncollateralized overnight call-rate target (policy rate).
# Source: Bank of Japan, "Chronology of monetary policy" (boj.or.jp); NIRP -0.10% Jan-2016;
# exit Mar-2024 (0-0.1%), 0.25% Jul-2024, 0.50% Jan-2025. Values in percent per year.
JP_RATE_STEPS = [
    ("2000-01-01", 0.00),   # ZIRP / QE era
    ("2006-07-14", 0.25),   # ZIRP exit
    ("2007-02-21", 0.50),
    ("2008-10-31", 0.30),   # GFC cuts
    ("2008-12-19", 0.10),
    ("2010-10-05", 0.05),   # 0-0.1% corridor ("comprehensive easing")
    ("2016-01-29", -0.10),  # NIRP
    ("2024-03-19", 0.05),   # NIRP exit, 0-0.1%
    ("2024-07-31", 0.25),
    ("2025-01-24", 0.50),
]

# ECB — deposit facility rate (the binding overnight rate under excess liquidity; ~ESTR).
# Source: ECB, "Key ECB interest rates" (ecb.europa.eu). Values in percent per year.
EUR_RATE_STEPS = [
    ("2011-12-14", 0.25),
    ("2012-07-11", 0.00),
    ("2014-06-11", -0.10),
    ("2014-09-10", -0.20),
    ("2015-12-09", -0.30),
    ("2016-03-16", -0.40),
    ("2019-09-18", -0.50),
    ("2022-07-27", 0.00),
    ("2022-09-14", 0.75),
    ("2022-11-02", 1.50),
    ("2022-12-21", 2.00),
    ("2023-02-08", 2.50),
    ("2023-03-22", 3.00),
    ("2023-05-10", 3.25),
    ("2023-06-21", 3.50),
    ("2023-09-20", 4.00),
    ("2024-06-12", 3.75),
    ("2024-09-18", 3.50),
    ("2024-10-23", 3.25),
    ("2024-12-18", 3.00),
    ("2025-02-05", 2.75),
    ("2025-03-12", 2.50),
    ("2025-04-23", 2.25),
    ("2025-06-11", 2.00),
]


def _step_series(steps: list[tuple[str, float]], index: pd.DatetimeIndex) -> pd.Series:
    """Turn a (date, level) step table into a series on ``index`` (last step at/before)."""
    dates = pd.DatetimeIndex([d for d, _ in steps])
    vals = np.array([v for _, v in steps], dtype=float)
    pos = dates.searchsorted(index, side="right") - 1
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], vals[0])
    return pd.Series(out, index=index)


def foreign_rate(ccy: str, index: pd.DatetimeIndex) -> pd.Series:
    """Hardcoded policy short rate (annual %) for 'JPY' or 'EUR' on ``index``."""
    steps = JP_RATE_STEPS if ccy == "JPY" else EUR_RATE_STEPS
    return _step_series(steps, index)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = "2005-01-01", end: str | None = None,
               path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download ETF total-return closes + spot FX + ^IRX and cache them (network, run once)."""
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide close frame (ETF total-return closes, FX spots, ^IRX %), cache-first."""
    if not os.path.exists(path):
        return fetch_tape(path=path)
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def monthly_panel(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Monthly panel: ETF simple returns, FX returns (value of the FOREIGN currency in USD),
    and annualized short rates (%): us_rate, jp_rate, eur_rate, plus rate diffs diff_JPY /
    diff_EUR (US minus foreign). Sliced to the as-of month-end (no partial months).
    """
    px = prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    m = m[m.index <= pd.Timestamp(asof)]

    out = pd.DataFrame(index=m.index)
    for etf in ["DXJ", "EWJ", "HEWJ", "HEDJ", "VGK"]:
        out[etf] = m[etf].pct_change()
    # JPY=X quotes yen per dollar -> the *yen's* USD return is the inverse ratio.
    out["fx_JPY"] = m["JPY=X"].shift(1) / m["JPY=X"] - 1.0
    # EURUSD=X quotes dollars per euro -> the euro's USD return is the plain ratio.
    out["fx_EUR"] = m["EURUSD=X"] / m["EURUSD=X"].shift(1) - 1.0
    out["us_rate"] = m["^IRX"]                          # annual %, 13-week bill
    out["jp_rate"] = foreign_rate("JPY", out.index)
    out["eur_rate"] = foreign_rate("EUR", out.index)
    out["diff_JPY"] = out["us_rate"] - out["jp_rate"]   # annual %
    out["diff_EUR"] = out["us_rate"] - out["eur_rate"]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 240, carry_annual: float = 0.05, seed: int = 613,
                    local_vol: float = 0.045, fx_vol: float = 0.028,
                    track_vol: float = 0.004) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED hedge carry.

    local equity return ~ N(0.5%, local_vol); fx return ~ N(0, fx_vol);
    unhedged = local + fx + tracking noise; hedged = local + carry_annual/12 + tracking noise.
    So (hedged - unhedged) + fx == carry/12 + noise, exactly the identity the estimator targets.
    ``carry_annual = 0`` is the null: the carry estimator must NOT manufacture significance.
    Decorative monthly index built with period_range (well under the 250-year ns-Timestamp cap).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2006-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    local = rng.normal(0.005, local_vol, n_months)
    fx = rng.normal(0.0, fx_vol, n_months)
    eps_h = rng.normal(0.0, track_vol, n_months)
    eps_u = rng.normal(0.0, track_vol, n_months)
    hedged = local + carry_annual / 12.0 + eps_h
    unhedged = local + fx + eps_u
    return pd.DataFrame(
        {"hedged": hedged, "unhedged": unhedged, "fx": fx,
         "rate_diff": np.full(n_months, carry_annual * 100.0)},  # annual %
        index=idx,
    )
