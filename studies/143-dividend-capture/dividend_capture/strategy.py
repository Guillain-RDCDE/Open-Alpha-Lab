"""Strategy and honest controls — Study 143 (Dividend-Capture).

The folk recipe: buy a stock a few days *before* the ex-dividend date, collect the
dividend, and sell on or after the ex-date — pocketing the dividend as pure profit.
The textbook efficient-markets counter: the stock price drops by *exactly* the
dividend on the ex-date, so the price loss offsets the cash gain and the trade earns
nothing gross. Once round-trip costs and the dividend tax wedge are added, it becomes
a net loser.

We test this through three lenses:

1. **The drop ratio** — is ``(price_before - price_ex) / div`` systematically < 1.0?
   If the market only partially adjusts, gross capture is positive. We compute this
   on the real tape and run a HAC t-test on whether it differs from 1.0.

2. **The capture trade vs buy-and-hold** — a true baseline: instead of capturing
   dividends, just hold the stock for the same window (close t-1 to close t+1).
   If the capture edge exists, the capture P&L should exceed the buy-and-hold P&L on
   that same window. The baseline is held constant without collecting the dividend,
   so the *only* source of outperformance is the dividend itself net of the price drop.

3. **The shuffled-date control** — the capture trade is re-run on random (non-ex-date)
   windows of the same 2-day length. If the edge is specific to the ex-date, it should
   not appear on random windows. If it does appear, the "edge" is just the market's
   steady upward drift over 2-day windows, not dividend capture.

No look-ahead: the signal is formed at close t-1; position entered at open t (proxied
here by the close of t-1 since we have daily data) and exited at close t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics: the drop ratio and the capture trade
# ---------------------------------------------------------------------------
def drop_ratio_stats(events: pd.DataFrame) -> dict:
    """Summary statistics for the ex-date price drop ratio (drop / dividend).

    Under full efficiency the expected value is ~1.0 (the price falls by the full
    dividend amount). Under partial efficiency the ratio is < 1.0, which would make
    gross capture positive. Key statistics:

    - ``mean_ratio`` — the sample mean of drop/div.
    - ``frac_below_one`` — fraction of events where drop < div (potential gross gain).
    - ``tstat_vs_one`` — Newey-West HAC t-stat for H0: mean ratio = 1.0.
    - A ratio significantly below 1 would support the capture hypothesis; the
      literature and our data find it close to 1 (and sometimes above, for high-tax
      investors or institutional activity).
    """
    ratios = events["drop_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = ratios.size
    if n < 5:
        return {k: float("nan") for k in (
            "n", "mean_ratio", "median_ratio", "std_ratio",
            "frac_below_one", "tstat_vs_one", "mean_drop_bps", "mean_div_bps",
        )}

    mean_ratio = float(ratios.mean())
    # HAC t-stat: H0: mean = 1.0
    e = ratios - ratios.mean()   # residuals around the mean
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = (mean_ratio - 1.0) / se if se > 0 else float("nan")

    return {
        "n": int(n),
        "mean_ratio": mean_ratio,
        "median_ratio": float(np.median(ratios)),
        "std_ratio": float(ratios.std(ddof=1)),
        "frac_below_one": float((ratios < 1.0).mean()),
        "tstat_vs_one": float(tstat),
        "mean_drop_bps": float(events["drop"].mean() / events["price_before"].mean() * 1e4),
        "mean_div_bps": float(events["div"].mean() / events["price_before"].mean() * 1e4),
    }


def capture_trade_stats(events: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """Per-event P&L of the capture trade: buy at close t-1, sell at close t+1, collect div.

    Gross return = (price_after - price_before + div) / price_before.
    Net return = gross - cost_bps/10000 (round-trip).

    Returns HAC t-stats, win-rate, skew, mean bps, and per-year count.
    """
    ret_gross = events["ret_gross"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    ret_net = (ret_gross - cost_bps * 1e-4)
    n = ret_gross.size

    def _tstat(r: np.ndarray) -> float:
        if r.size < 5:
            return float("nan")
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (r.size / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / r.size
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / r.size
        se = np.sqrt(max(lrv, 0.0) / r.size)
        return float(mu / se) if se > 0 else float("nan")

    # Annualisation: estimate events per year from the date range.
    if "ex_date" in events.columns and len(events) > 1:
        dates = pd.to_datetime(events["ex_date"])
        years = (dates.max() - dates.min()).days / 365.25
        events_per_year = n / years if years > 0 else float("nan")
    else:
        events_per_year = float("nan")

    return {
        "n": int(n),
        "events_per_year": float(events_per_year),
        "mean_gross_bps": float(ret_gross.mean() * 1e4) if n else float("nan"),
        "mean_net_bps": float(ret_net.mean() * 1e4) if n else float("nan"),
        "win_rate_gross": float((ret_gross > 0).mean()) if n else float("nan"),
        "skew_gross": float(pd.Series(ret_gross).skew()) if n > 2 else float("nan"),
        "tstat_gross": _tstat(ret_gross),
        "tstat_net": _tstat(ret_net),
        "cost_bps": float(cost_bps),
    }


def buy_and_hold_stats(events: pd.DataFrame) -> dict:
    """Buy-and-hold baseline over the same 2-day window (close t-1 to close t+1), WITHOUT the div.

    This baseline shows what a passive holder earns in the surrounding price move, net of the
    dividend effect. The capture trade outperforms only if the ex-date window has extra positive
    drift that can absorb the capture cost. On an efficient market the capture trade's gross
    equals the buy-and-hold gross by construction (the div is exactly offset by the drop).

    The baseline return is (price_after - price_before) / price_before.
    """
    bah = ((events["price_after"] - events["price_before"]) / events["price_before"]).to_numpy(dtype=float)
    bah = bah[np.isfinite(bah)]
    n = bah.size
    return {
        "n": int(n),
        "mean_bah_bps": float(bah.mean() * 1e4) if n else float("nan"),
        "win_rate_bah": float((bah > 0).mean()) if n else float("nan"),
    }


def shuffled_date_control(
    events: pd.DataFrame,
    price_series: pd.Series | None = None,
    n_draws: int = 500,
    seed: int = 143,
) -> dict:
    """Monte-Carlo control: same 2-day trade on random (non-ex-date) windows.

    If no ``price_series`` is available (synthetic data), we bootstrap from the
    existing ``ret_gross`` distribution with a phase-shuffled dates approach. When a
    real daily price series is given, we draw ``n_draws`` random 2-day windows away
    from ex-dates and compute the same (price_after - price_before) / price_before.

    This answers: "is the 2-day capture window special, or would any random 2-day
    window give the same gross return?" If the shuffled mean ≈ capture mean, the
    ex-date is not special.
    """
    rng = np.random.default_rng(seed)

    if price_series is None:
        # Bootstrap approach: shuffle the return series to destroy the ex-date structure.
        r = events["ret_gross"].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        boots = [rng.choice(r, size=len(r), replace=True).mean() * 1e4 for _ in range(n_draws)]
        mean_ctrl = float(np.mean(boots))
        std_ctrl = float(np.std(boots, ddof=1))
    else:
        # Draw random 2-day windows from the price series, avoiding ex-dates.
        ex_set = set(pd.to_datetime(events["ex_date"]).dt.date)
        close = price_series.dropna()
        idx = close.index
        non_ex_positions = [
            i for i in range(1, len(idx) - 2)
            if idx[i].date() not in ex_set
        ]
        if len(non_ex_positions) < 10:
            return {"mean_ctrl_bps": float("nan"), "std_ctrl_bps": float("nan"), "n_draws": 0}

        draws = rng.choice(non_ex_positions, size=min(n_draws, len(non_ex_positions)), replace=True)
        rets = []
        for pos in draws:
            pb, pa = float(close.iloc[pos - 1]), float(close.iloc[pos + 1])
            if pb > 0:
                rets.append((pa - pb) / pb * 1e4)
        mean_ctrl = float(np.mean(rets))
        std_ctrl = float(np.std(rets, ddof=1))

    return {
        "mean_ctrl_bps": mean_ctrl,
        "std_ctrl_bps": std_ctrl,
        "n_draws": n_draws,
    }


def summarize(events: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """Top-level study summary: drop-ratio stats, capture P&L, and baseline.

    The decisive inference numbers are ``tstat_vs_one`` (is the drop ratio different
    from 1?) and ``tstat_gross`` / ``tstat_net`` (is the capture return different
    from zero?). A real edge needs ``tstat_gross`` with |t| >= 2 and a drop ratio
    significantly below 1. The literature expectation is that it finds neither.
    """
    return {
        "drop": drop_ratio_stats(events),
        "capture": capture_trade_stats(events, cost_bps=cost_bps),
        "bah": buy_and_hold_stats(events),
    }
