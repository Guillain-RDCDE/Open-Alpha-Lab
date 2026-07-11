"""Data layer for Study 658 — Put-Write-Premium.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily auto-adjusted (total-return) closes for **PUTW** (WisdomTree CBOE S&P
  500 PutWrite Strategy Fund — the only liquid, investable wrapper on the CBOE PUT methodology;
  sells a rolling one-month at-the-money S&P 500 put, cash-secured with T-bills), **SPY** (the
  buy-and-hold benchmark it is pitched against) and **BIL** (1-3 month T-bill ETF — the cash
  leg / risk-free proxy every excess-of-cash Sharpe in this study is computed against, exactly
  the convention used by sibling study 655-ivy-portfolio). All from yfinance (no key), cached
  under the study's own ``_cache/`` as CSV, cache-first.

  PUTW inception is **2016-02-24** — the honest, *entire* live history of the only tradable
  product that actually implements this claim; the CBOE PUT INDEX itself goes back to 1986, but
  quoting the *index* (no fees, no tracking, no real fills) instead of the *fund* would be
  claiming a return nobody could have banked. We name that trade-off explicitly rather than
  hardcode a synthetic pre-2016 PUT-index proxy: a ~9.4-year, ~124-month sample is what the
  claim's own vehicle has actually delivered, single strong bull regime included and named.

* **Named crash windows (hardcoded facts, no network).** Two textbook equity-vol shocks that
  fall inside PUTW's live tape, used to check whether the (lower-beta) put-writer actually
  cushions a crash or just carries a smaller average beta that widens exactly when stressed:
  the **February 2018 "Volmageddon"** vol-spike (XIV terminated 2018-02-06) and the **2020
  COVID crash**. Dates are the conventional peak-to-trough windows cited for each episode.

* **Synthetic world.** A deterministic, seeded monthly cash-secured-ATM-put engine (Black-
  Scholes premium) with a TUNABLE **variance-risk-premium knob** (``harvest``): options are
  priced off ``sigma_realized * (1 + harvest)``. ``harvest = 0`` means the world's options are
  priced at exactly the realized vol that generates the underlying — a "fair" world with no
  volatility risk premium embedded, so a CAPM-alpha regression run on the engine's monthly
  paper should NOT find significant alpha. ``harvest > 0`` plants a genuine embedded premium
  (implied systematically richer than realized) that the same regression must recover. This is
  a faithful-engine / power check on the *alpha detector*, never evidence for the real-tape
  stamp.

Pure numpy + pandas + scipy.stats.norm on the offline path; ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "pwp_prices.csv")

PUTW = "PUTW"    # WisdomTree CBOE S&P 500 PutWrite Strategy Fund, inception 2016-02-24
SPY = "SPY"      # buy-and-hold benchmark
CASH = "BIL"     # T-bill ETF: cash leg + risk-free proxy for every excess-of-cash Sharpe
TICKERS = [PUTW, SPY, CASH]

START = "2016-02-24"       # PUTW inception — the entire live history of the tradable product
AS_OF = "2026-06-30"       # last complete month at publication (2026-07-10 run date)

# --------------------------------------------------------------------------- #
# Named crash windows (hardcoded facts, no network) — conventional peak-to-trough dates for
# two vol shocks that fall inside PUTW's live tape.
# --------------------------------------------------------------------------- #
CRASH_WINDOWS = {
    "volmageddon_2018": ("2018-01-26", "2018-02-09"),  # SPY local peak -> post-XIV-death trough
    "covid_2020": ("2020-02-19", "2020-03-23"),          # SPY all-time-high -> COVID-crash trough
}

# A large-single-day-down threshold used for the crash-conditional beta test below — an
# objective, ex-ante-definable cut (not a full-sample quantile), so the "beta widens exactly
# when it hurts" test isn't snooped on the sample it's run on.
CRASH_DAY_THRESHOLD = -0.03   # SPY daily total-return <= -3%


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2015-06-01", end: str = "2026-07-05") -> None:
    """Download PUTW / SPY / BIL auto-adjusted (total-return) daily closes; cache. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    px = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px[TICKERS].dropna(how="all")
    px.to_csv(PRICES_CACHE)


def have_real() -> bool:
    return os.path.exists(PRICES_CACHE)


def load_prices() -> pd.DataFrame:
    return pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True).sort_index()[TICKERS]


def load_real(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached PUTW/SPY/BIL close panel, sliced to [start, asof] with all-3-present rows only
    (the joint window is bound by PUTW's 2016-02-24 inception, the binding constraint)."""
    px = load_prices()
    px = px.loc[(px.index >= start) & (px.index <= asof)].dropna(how="any")
    return px


# --------------------------------------------------------------------------- #
# Synthetic world — cash-secured ATM put-write, planted variance-risk-premium knob
# --------------------------------------------------------------------------- #
def bs_atm_put(sigma_ann: np.ndarray | float, t_years: float = 1.0 / 12.0) -> np.ndarray:
    """Black-Scholes price of a 1-month at-the-money (K = S) put, as a fraction of spot.

    ``r = 0`` (near-zero real carry over a one-month horizon):
    ``d1 = sigma*sqrt(t)/2``, ``d2 = -d1``, ``put/S = Phi(-d2) - Phi(-d1)``.
    """
    from scipy.stats import norm

    sigma = np.asarray(sigma_ann, dtype=float)
    srt = np.maximum(sigma * np.sqrt(t_years), 1e-9)
    d1 = 0.5 * srt
    d2 = -d1
    return norm.cdf(-d2) - norm.cdf(-d1)


def synthetic_world(harvest: float = 0.0, seed: int = 658, n_months: int = 240,
                    sigma_ann: float = 0.16, mu_ann: float = 0.09,
                    cash_ann: float = 0.018) -> pd.DataFrame:
    """Deterministic monthly synthetic panel: lognormal SPY-like underlying, a cash-secured
    ATM put-write engine priced at ``sigma_ann * (1 + harvest)``, and a flat cash leg.

    ``harvest = 0``: options are priced at exactly the realized vol that generates the
    underlying — no embedded volatility risk premium; a CAPM-alpha regression of the put-write
    engine on the underlying must NOT find significant alpha (the null the detector must
    respect). ``harvest > 0``: implied is systematically richer than realized by that fraction
    (the textbook VRP) — the same regression must recover a significant, positive alpha.

    20 years of monthly data (well under the ~3,000-point / 250-year ns-Timestamp ceiling);
    built on a ``period_range`` to sidestep the trap outright. Columns: ``spy_ret``,
    ``putw_ret``, ``cash_ret``.
    """
    if n_months > 3_000:
        raise ValueError("keep the synthetic span under 250 years")
    rng = np.random.default_rng(seed)
    mu_m = mu_ann / 12.0
    sig_m = sigma_ann / np.sqrt(12.0)
    z = rng.standard_normal(n_months)
    spy_ret = np.exp(mu_m - 0.5 * sig_m ** 2 + sig_m * z) - 1.0

    sigma_iv = sigma_ann * (1.0 + harvest)
    p = float(bs_atm_put(sigma_iv))              # constant monthly ATM premium, % of spot
    # cash-secured put, strike = spot, rebased every month: expires worthless (keep premium)
    # if the month is flat/up; assigned (eat the drop, keep premium) if the month is down.
    putw_ret = np.where(spy_ret >= 0.0, p, spy_ret + p)

    cash_ret = np.full(n_months, cash_ann / 12.0)

    idx = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp(how="end")
    return pd.DataFrame({"spy_ret": spy_ret, "putw_ret": putw_ret, "cash_ret": cash_ret},
                        index=idx)
