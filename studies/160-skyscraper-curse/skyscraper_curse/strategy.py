"""Strategy and honest controls — Study 160 (Skyscraper-Curse).

The folk recipe (Lawrence 1999): record-tall building completions cluster near market
peaks and precede crashes. We test this as a forward-return event study:

- **Event window**: S&P 500 total returns in the 1, 2, and 3 calendar years AFTER each
  record completion year.
- **Unconditional baseline**: the full-sample mean annual S&P 500 return over the same
  historical period — the simplest fair comparison.
- **Random-day control**: draw 2,000 random subsets of the same n from the annual
  return series, compute the mean forward return for each, and compare the observed
  event mean against that empirical distribution. This answers "is the event timing
  special, or just average BTC-era returns rebranded?"
- **Drawdown analysis**: the maximum drawdown observed in the 3 years after each event,
  versus the unconditional 3-year max-drawdown distribution.

The honest teardown: n ≈ 6–9 world-record completions fall within the S&P 500 era.
A big mean and even a big t-stat on n=6 is expected from sampling variance alone — the
power floor is far below the inference bar. We show the power calculation explicitly.

No look-ahead: the completion year is the event; the forward window starts the
following calendar year (January of year+1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shiller monthly → annual total return
# ---------------------------------------------------------------------------
def shiller_annual_returns(shiller: pd.DataFrame, real: bool = False) -> pd.Series:
    """Convert the Shiller monthly frame to annual total log-returns (S&P 500).

    Uses the SP500 price index and Dividend column to compute total return (price +
    reinvested dividends). With ``real=True``, deflates by the Consumer Price Index
    (Shiller CPI column) to get real total returns.

    The Shiller frame is expected to have a DatetimeIndex (as returned by
    ``data.load_shiller()``). Returns a pd.Series indexed by integer year.
    """
    df = shiller.copy()
    # Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Monthly dividend yield: Dividend / 12 / SP500 (Shiller's Dividend is annualised)
    df["div_yield_monthly"] = df["Dividend"] / 12.0 / df["SP500"]
    # Monthly total return = price return + dividend reinvested
    df["price_ret"] = df["SP500"] / df["SP500"].shift(1) - 1.0
    df["total_ret"] = (1 + df["price_ret"]) * (1 + df["div_yield_monthly"].shift(1).fillna(0)) - 1.0

    if real:
        cpi_col = [c for c in df.columns if "Consumer Price Index" in str(c) or c == "CPI"]
        if cpi_col:
            cpi = df[cpi_col[0]]
            df["cpi_ret"] = cpi / cpi.shift(1) - 1.0
            df["total_ret"] = (1 + df["total_ret"]) / (1 + df["cpi_ret"].fillna(0)) - 1.0

    df["year"] = df.index.year
    # Annual total return: compound monthly returns
    annual = (
        df.groupby("year")["total_ret"]
        .apply(lambda r: (1 + r).prod() - 1)
        .rename("annual_total_ret")
    )
    return annual


# ---------------------------------------------------------------------------
# Core event-window study
# ---------------------------------------------------------------------------
def event_forward_returns(
    annual_rets: pd.Series,
    event_years: list[int],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Forward S&P total returns starting the year AFTER each event year.

    For each event year ``e`` and each horizon ``h`` (years), compute the compound
    annual total return from (e+1) through (e+h) inclusive. Entry is the first
    calendar year after the event — no look-ahead.

    Returns a DataFrame with one row per (event, horizon) combination.
    Columns: event_year, horizon, start_year, end_year, compound_ret, annualised_ret,
             n_years_actual.
    """
    if horizons is None:
        horizons = [1, 2, 3]

    rows = []
    for ey in event_years:
        for h in horizons:
            start = ey + 1
            end = ey + h
            window = annual_rets.loc[
                (annual_rets.index >= start) & (annual_rets.index <= end)
            ]
            if len(window) == 0:
                continue
            compound = (1 + window).prod() - 1.0
            n_act = len(window)
            ann = (1 + compound) ** (1.0 / n_act) - 1.0 if n_act > 0 else float("nan")
            rows.append(
                {
                    "event_year": ey,
                    "horizon": h,
                    "start_year": start,
                    "end_year": end,
                    "compound_ret": compound,
                    "annualised_ret": ann,
                    "n_years_actual": n_act,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Random-day control — empirical p-value
# ---------------------------------------------------------------------------
def random_control(
    annual_rets: pd.Series,
    n_events: int,
    horizon: int = 1,
    n_draws: int = 2000,
    seed: int = 160,
) -> pd.Series:
    """Monte-Carlo control: draw ``n_draws`` random bundles of ``n_events`` starting
    years (each having at least ``horizon`` years of forward data), compute the mean
    compound forward return, and return the distribution.

    This answers: "how often would a random bundle of equally-sized event windows
    produce a mean return as extreme as the skyscraper events?"
    """
    rng = np.random.default_rng(seed)
    min_year = int(annual_rets.index.min())
    max_year = int(annual_rets.index.max()) - horizon

    eligible = [y for y in annual_rets.index if min_year <= y <= max_year]
    if len(eligible) < n_events:
        return pd.Series(dtype=float)

    means = []
    for _ in range(n_draws):
        chosen = rng.choice(eligible, size=n_events, replace=False)
        rets = []
        for ey in chosen:
            window = annual_rets.loc[
                (annual_rets.index >= ey + 1) & (annual_rets.index <= ey + horizon)
            ]
            if len(window) > 0:
                compound = (1 + window).prod() - 1.0
                rets.append(compound)
        if rets:
            means.append(float(np.mean(rets)))
    return pd.Series(means, name="random_mean_fwd_ret")


# ---------------------------------------------------------------------------
# Summarise event windows with HAC t-stat
# ---------------------------------------------------------------------------
def summarize(returns: np.ndarray | pd.Series, label: str = "") -> dict:
    """Headline statistics for an array of event forward returns.

    Returns n, mean, std, min, max, HAC t-stat (Newey-West), and a low-power flag.
    With n <= 10, the t-stat is explicitly unreliable; we compute it anyway and flag it.

    This is the inference-bar function: with n ≈ 6–9 events, |t| >= 2 is nearly
    impossible to reach without effect sizes that dwarf the unconditional baseline —
    and even then, multiple-event rarity means the claim belongs in 'narrative' not
    'signal'.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out: dict = {
        "label": label,
        "n": int(n),
        "mean": float(r.mean()) if n else float("nan"),
        "std": float(r.std(ddof=1)) if n > 1 else float("nan"),
        "min": float(r.min()) if n else float("nan"),
        "max": float(r.max()) if n else float("nan"),
        "tstat": float("nan"),
        "low_power_warning": bool(n < 15),
    }
    if n > 1:
        mu = r.mean()
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, min(lags + 1, n)):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Unconditional baseline
# ---------------------------------------------------------------------------
def unconditional_stats(annual_rets: pd.Series, horizon: int = 1) -> dict:
    """Compute the unconditional mean h-year compound forward return for all
    overlapping h-year windows in the sample. This is the fairest baseline: what
    any randomly chosen h-year stretch of the tape typically returns.
    """
    compound_all = []
    ys = sorted(annual_rets.index)
    for i, y in enumerate(ys):
        window = annual_rets.loc[
            (annual_rets.index >= y) & (annual_rets.index < y + horizon)
        ]
        if len(window) == horizon:
            compound_all.append(float((1 + window).prod() - 1.0))
    return summarize(np.array(compound_all), label=f"unconditional_{horizon}yr")


# ---------------------------------------------------------------------------
# Power analysis — how many events do we need?
# ---------------------------------------------------------------------------
def power_at_n(
    annual_rets: pd.Series,
    n_events_range: list[int] | None = None,
    horizon: int = 1,
    n_draws: int = 2000,
    seed: int = 160,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Estimate statistical power as a function of event count, using a bootstrap
    procedure.

    For each n in ``n_events_range``, draw ``n_draws`` random bundles of n years from
    the tape, compute the mean forward return, and count how often it exceeds the
    (1-alpha) quantile of the full random distribution. This is the fraction of the
    time n random events would look 'significant' — a calibration of the bar.

    With the true effect size unknown, we use the observed event mean as a proxy.
    The result is a DataFrame with columns [n, power_estimate].
    """
    if n_events_range is None:
        n_events_range = [3, 5, 6, 9, 15, 30, 60]

    ctrl_full = random_control(annual_rets, n_events=30, horizon=horizon,
                               n_draws=1000, seed=seed)
    threshold = float(ctrl_full.quantile(1 - alpha)) if len(ctrl_full) else float("nan")

    rows = []
    rng = np.random.default_rng(seed + 1)
    min_year = int(annual_rets.index.min())
    max_year = int(annual_rets.index.max()) - horizon
    eligible = [y for y in annual_rets.index if min_year <= y <= max_year]

    for n in n_events_range:
        if n > len(eligible):
            continue
        exc = 0
        for _ in range(n_draws):
            chosen = rng.choice(eligible, size=n, replace=False)
            rets = []
            for ey in chosen:
                window = annual_rets.loc[
                    (annual_rets.index >= ey + 1) & (annual_rets.index <= ey + horizon)
                ]
                if len(window) > 0:
                    rets.append(float((1 + window).prod() - 1.0))
            if rets and np.mean(rets) > threshold:
                exc += 1
        rows.append({"n": n, "power_estimate": exc / n_draws, "threshold": threshold})
    return pd.DataFrame(rows)
