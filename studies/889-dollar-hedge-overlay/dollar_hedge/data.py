"""Data layer for Study 889 — Broad Dollar-Hedge Overlay.

**The claim under test.** Study 613 showed that for ONE market (Japan) the return gap between a
currency-*hedged* equity ETF and its *unhedged* twin is, mechanically, the covered-interest-parity
short-rate differential — "free carry hidden in a share class". Here we **generalise to broad
developed international** (MSCI EAFE): is the hedged-minus-unhedged return the (now positive,
dollar-favourable) US-vs-foreign rate differential, mechanically? And does a systematic *"hedge
when the US out-yields"* overlay add Sharpe once it is costed?

A currency-hedged EAFE fund sells the foreign-currency basket forward; CIP prices that forward at
the short-rate differential, so per month (log-approx):

    hedged   ~ local_equity + carry            carry = (r_US - r_foreign_basket)/12
    unhedged ~ local_equity + fx_foreign        fx_foreign = USD return of the foreign basket
    =>  diff := hedged - unhedged  ~  carry - fx_foreign  =  carry + dollar_return

because the foreign basket's USD return is (minus) the broad-dollar return. Rearranged, the hedge
carry estimate is ``carry_hat := diff + fx_foreign = diff - dollar_return``.

**The ETF pairs (yfinance total-return closes).**

  * **HEFA vs EFA** (2014-03+) — iShares Currency Hedged MSCI EAFE literally *holds EFA plus
    one-month currency forwards*: the **same basket**, so the return differential is the hedge P&L
    almost purely. The clean mechanical pair (the broad analogue of 613's HEWJ/EWJ).
  * **DBEF vs EFA** (2011-06+) — Xtrackers MSCI EAFE Hedged Equity vs iShares MSCI EAFE. Same MSCI
    EAFE index, different providers/optimised baskets → a little basket noise, but a much longer
    tape spanning the whole US-under-yields → US-out-yields regime flip.
  * **IEFA** (iShares Core MSCI EAFE, 2012-10+) is carried as a near-identical *unhedged* leg for a
    robustness check that the choice of unhedged wrapper barely moves the result.

**The dollar leg.** ``UUP`` (Invesco DB US Dollar Index Bullish) tracks the DXY basket (EUR 57.6%,
JPY 13.6%, GBP 11.9%, CAD/SEK/CHF). The EAFE currency basket (EUR ~33%, JPY ~22%, GBP ~16%,
CHF ~10%, AUD ~9%) overlaps the DXY strongly but not perfectly, so ``dollar_return = UUP`` is a
**documented proxy** for the broad dollar; the residual basket mismatch is honest noise on
``carry_hat`` and is why HEFA/EFA (same equity basket) is the decisive pair.

**Rates.** US short rate = ``^IRX`` (13-week T-bill discount, %). The foreign short rate is a
coarse **EAFE-weighted policy-rate blend** (ECB 0.40, BOJ 0.25, BOE 0.20, SNB 0.15) built from
hardcoded step tables (sources cited below). Coarse (±~20 bp vs the true basket bill), but the
differential story lives at the 1-5 %/yr scale, far above that tolerance. ``BIL`` (1-3m T-bill
ETF) is the cash leg for excess-of-cash Sharpe races.

**Synthetic world.** A deterministic monthly generator with a TUNABLE planted carry (knob
``carry_annual``) plus an fx/dollar leg and basket noise — the positive control (and, at
``carry_annual = 0``, the null that must NOT fire). An optional ``flip_half`` regime plants carry
only in the second half (US out-yields) so the "hedge when the US out-yields" overlay has a real
regime to switch on. Index built with ``period_range`` (spans well under 250 years).

Cache-first: ``fetch`` (network, yfinance) runs once and writes ``_cache/dh_prices.csv``;
everything else is offline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(CACHE_DIR, "dh_prices.csv")

# ETF total-return closes + dollar proxy + US bill + FX spot basket. UUP = DXY basket bullish fund
# (a TRADEABLE dollar whose total return also earns the US-bill collateral yield — see below);
# ^IRX = 13-week T-bill discount (%); BIL = 1-3m T-bill ETF (cash leg); the four FX spots build the
# clean EAFE currency basket used for the carry identity (spot has NO collateral yield, unlike UUP).
ETF_TICKERS = ["HEFA", "EFA", "DBEF", "IEFA", "UUP", "BIL", "^IRX"]
FX_TICKERS = ["EURUSD=X", "GBPUSD=X", "JPY=X", "CHF=X"]
TICKERS = ETF_TICKERS + FX_TICKERS

# EAFE-weighted currency basket for the *spot* fx leg (matches the FOREIGN_BLEND rate weights).
# EURUSD=X / GBPUSD=X quote USD-per-unit (currency's USD return = plain ratio); JPY=X / CHF=X quote
# units-per-USD (currency's USD return = the INVERSE ratio).
FX_WEIGHTS = {"EUR": 0.40, "JPY": 0.25, "GBP": 0.20, "CHF": 0.15}

# WHY spot and not UUP for the carry identity. The identity is diff = carry - fx_foreign, so
# carry_hat = diff + fx_foreign needs the pure SPOT return of the foreign basket. UUP's TOTAL
# return ~ dollar_spot + US-bill_collateral_yield - fees, so naively using carry_hat = diff - UUP
# subtracts the collateral yield too and cancels most of the carry (it prints NEGATIVE, worst in the
# high-rate 2022+ era) — a documented trap kept in the notebook. The spot basket avoids it.

# (hedged, unhedged, label). HEFA/EFA is the same-basket clean pair; DBEF/EFA the longer, slightly
# basket-mismatched pair. HEFA listed 2014-01 (first clean month 2014-03); DBEF 2011-06.
PAIRS = {
    "HEFA/EFA": ("HEFA", "EFA", "EAFE 2014+ (SAME basket: HEFA holds EFA + one-month FX forwards)"),
    "DBEF/EFA": ("DBEF", "EFA", "EAFE 2011+ (same index, different provider/basket)"),
}

AS_OF = "2026-06-30"  # last complete calendar month at build time; the partial current month drops.

# --------------------------------------------------------------------------- #
# Hardcoded foreign short-rate step tables (annual %, effective from date)
# --------------------------------------------------------------------------- #
# ECB deposit facility rate (the binding overnight rate under excess liquidity).
# Source: ECB, "Key ECB interest rates" (ecb.europa.eu).
ECB_STEPS = [
    ("2011-12-14", 0.25), ("2012-07-11", 0.00), ("2014-06-11", -0.10), ("2014-09-10", -0.20),
    ("2015-12-09", -0.30), ("2016-03-16", -0.40), ("2019-09-18", -0.50), ("2022-07-27", 0.00),
    ("2022-09-14", 0.75), ("2022-11-02", 1.50), ("2022-12-21", 2.00), ("2023-02-08", 2.50),
    ("2023-03-22", 3.00), ("2023-05-10", 3.25), ("2023-06-21", 3.50), ("2023-09-20", 4.00),
    ("2024-06-12", 3.75), ("2024-09-18", 3.50), ("2024-10-23", 3.25), ("2024-12-18", 3.00),
    ("2025-02-05", 2.75), ("2025-03-12", 2.50), ("2025-04-23", 2.25), ("2025-06-11", 2.00),
]
# Bank of Japan uncollateralized overnight call-rate target.
# Source: Bank of Japan, "Chronology of monetary policy" (boj.or.jp).
BOJ_STEPS = [
    ("2008-12-19", 0.10), ("2010-10-05", 0.05), ("2016-01-29", -0.10), ("2024-03-19", 0.05),
    ("2024-07-31", 0.25), ("2025-01-24", 0.50),
]
# Bank of England Bank Rate. Source: Bank of England, "Official Bank Rate history" (bankofengland.co.uk).
BOE_STEPS = [
    ("2009-03-05", 0.50), ("2016-08-04", 0.25), ("2017-11-02", 0.50), ("2018-08-02", 0.75),
    ("2020-03-11", 0.25), ("2020-03-19", 0.10), ("2021-12-16", 0.25), ("2022-02-03", 0.50),
    ("2022-03-17", 0.75), ("2022-05-05", 1.00), ("2022-06-16", 1.25), ("2022-08-04", 1.75),
    ("2022-09-22", 2.25), ("2022-11-03", 3.00), ("2022-12-15", 3.50), ("2023-02-02", 4.00),
    ("2023-03-23", 4.25), ("2023-05-11", 4.50), ("2023-06-22", 5.00), ("2023-08-03", 5.25),
    ("2024-08-01", 5.00), ("2024-11-07", 4.75), ("2025-02-06", 4.50), ("2025-05-08", 4.25),
    ("2025-08-07", 4.00),
]
# Swiss National Bank policy rate. Source: SNB, "Current interest rates and exchange rates" (snb.ch).
SNB_STEPS = [
    ("2011-08-03", 0.00), ("2015-01-15", -0.75), ("2022-06-16", -0.25), ("2022-09-22", 0.50),
    ("2022-12-15", 1.00), ("2023-03-23", 1.50), ("2023-06-22", 1.75), ("2024-03-21", 1.50),
    ("2024-06-20", 1.25), ("2024-09-26", 1.00), ("2024-12-12", 0.50), ("2025-03-20", 0.25),
    ("2025-06-19", 0.00),
]

# EAFE-weighted blend of the four dominant developed-ex-US currency blocks.
FOREIGN_BLEND = {"ECB": 0.40, "BOJ": 0.25, "BOE": 0.20, "SNB": 0.15}
_STEP_TABLES = {"ECB": ECB_STEPS, "BOJ": BOJ_STEPS, "BOE": BOE_STEPS, "SNB": SNB_STEPS}


def _step_series(steps: list[tuple[str, float]], index: pd.DatetimeIndex) -> pd.Series:
    """Turn a (date, level) step table into a series on ``index`` (last step at/before)."""
    dates = pd.DatetimeIndex([d for d, _ in steps])
    vals = np.array([v for _, v in steps], dtype=float)
    pos = dates.searchsorted(index, side="right") - 1
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], vals[0])
    return pd.Series(out, index=index)


def foreign_rate(index: pd.DatetimeIndex) -> pd.Series:
    """EAFE-weighted foreign policy short rate (annual %) on ``index`` (ECB/BOJ/BOE/SNB blend)."""
    out = pd.Series(0.0, index=index)
    for bank, w in FOREIGN_BLEND.items():
        out = out + w * _step_series(_STEP_TABLES[bank], index)
    return out


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2011-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download ETF total-return closes + UUP + ^IRX and cache them (network, run once)."""
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
    if raw is None:
        raise RuntimeError("yfinance returned nothing after retries")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide close frame (ETF total-return closes, UUP, ^IRX %), cache-first."""
    if not os.path.exists(path):
        return fetch(path=path)
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def monthly_panel(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Monthly panel: ETF simple returns, dollar (UUP) return, cash (BIL) return, US/foreign short
    rates (annual %) and the differential. Sliced to the as-of month-end (no partial month)."""
    px = prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    m = m[m.index <= pd.Timestamp(asof)]

    out = pd.DataFrame(index=m.index)
    for etf in ["HEFA", "EFA", "DBEF", "IEFA", "BIL"]:
        out[etf] = m[etf].pct_change()
    out["dollar"] = m["UUP"].pct_change()      # TRADEABLE dollar (embeds collateral yield; see notes)
    out["fx_foreign"] = _fx_basket(m)          # SPOT USD return of the EAFE currency basket
    out["us_rate"] = m["^IRX"]                  # annual %, 13-week bill
    out["foreign_rate"] = foreign_rate(out.index)
    out["diff_rate"] = out["us_rate"] - out["foreign_rate"]  # annual %, US minus EAFE blend
    return out


def _fx_basket(m: pd.DataFrame) -> pd.Series:
    """EAFE-weighted SPOT USD return of the foreign currency basket from the four FX spots."""
    eur = m["EURUSD=X"].pct_change()                     # USD per EUR -> euro's USD return
    gbp = m["GBPUSD=X"].pct_change()                     # USD per GBP -> pound's USD return
    jpy = m["JPY=X"].shift(1) / m["JPY=X"] - 1.0         # USDJPY inverted -> yen's USD return
    chf = m["CHF=X"].shift(1) / m["CHF=X"] - 1.0         # USDCHF inverted -> franc's USD return
    return (FX_WEIGHTS["EUR"] * eur + FX_WEIGHTS["JPY"] * jpy
            + FX_WEIGHTS["GBP"] * gbp + FX_WEIGHTS["CHF"] * chf)


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 180, carry_annual: float = 0.03, seed: int = 889,
                    local_vol: float = 0.040, dollar_vol: float = 0.022,
                    track_vol: float = 0.004, flip_half: bool = False) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED hedge carry.

    ``dollar`` ~ N(0, dollar_vol) is the broad-dollar return; the foreign basket USD return is
    ``fx = -dollar``. local equity ~ N(0.4%, local_vol).

        unhedged = local + fx + tracking noise
        hedged   = local + carry/12 + tracking noise

    So ``diff = hedged - unhedged = carry/12 + dollar + noise`` and ``carry_hat = diff - dollar``
    recovers ``carry/12``, exactly the identity the estimator targets. ``carry_annual = 0`` is the
    null (the estimator must NOT manufacture significance).

    ``flip_half=True`` plants the carry (and a positive rate differential) only in the SECOND half
    of the sample, with zero carry and a negative differential in the first — a regime the
    "hedge when the US out-yields" overlay must switch on. Decorative monthly index via
    ``period_range`` (well under the ns-Timestamp cap).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2011-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    local = rng.normal(0.004, local_vol, n_months)
    dollar = rng.normal(0.0, dollar_vol, n_months)
    eps_h = rng.normal(0.0, track_vol, n_months)
    eps_u = rng.normal(0.0, track_vol, n_months)

    carry_m = np.full(n_months, carry_annual / 12.0)
    rate_diff = np.full(n_months, carry_annual * 100.0)
    if flip_half:
        half = n_months // 2
        carry_m[:half] = 0.0
        rate_diff[:half] = -1.0  # US under-yields in the first regime

    fx = -dollar  # the foreign basket's USD spot return is minus the dollar's
    hedged = local + carry_m + eps_h
    unhedged = local + fx + eps_u
    cash = np.full(n_months, 0.02 / 12.0)  # flat 2%/yr cash leg
    return pd.DataFrame(
        {"HEFA": hedged, "EFA": unhedged, "DBEF": hedged, "IEFA": unhedged,
         "BIL": cash, "dollar": dollar, "fx_foreign": fx,
         "us_rate": 2.0 + rate_diff, "foreign_rate": 2.0,
         "diff_rate": rate_diff},
        index=idx,
    )
