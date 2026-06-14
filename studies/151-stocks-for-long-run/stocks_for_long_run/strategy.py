"""Strategy and honest controls — Study 151 (Stocks-For-Long-Run).

The claim (Siegel's "Stocks for the Long Run"): *over long enough horizons
equities always deliver positive real returns and always beat bonds*.  We test
two versions of the claim against an honest null:

Claim A — No negative real equity return over 20/30-year windows:
    Metric: worst-case rolling N-year annualised real equity total return.
    Verdict criterion: is it above zero on every rolling window?

Claim B — Stocks beat bonds over 20/30-year windows:
    Metric: rolling N-year equity-minus-bond annualised excess return.
    Verdict criterion: is it positive on every rolling window?

Control (the "buy bonds" baseline):
    The rolling N-year real bond return under the same 10-year-yield proxy.
    This pins the claim "vs bonds" to an explicit counterfactual, not thin air.

HAC inference on the per-period (monthly-starting-window) excess return series
tells us whether the *mean* excess is statistically distinguishable from zero —
though at long horizons the sample is tiny (overlapping windows) so we also
report the empirical worst-case directly (no parametric assumption needed for
the "always beats" claim).

No look-ahead: every rolling window is computed on historical data up to the
end of that window.  The signal at the *start* of a holding period uses only
info available at that moment; the forward return is realised over the
subsequent N years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HORIZONS, load_shiller


# ---------------------------------------------------------------------------
# Rolling-window summary helpers
# ---------------------------------------------------------------------------
def worst_case_windows(df: pd.DataFrame, horizon: int) -> dict:
    """Worst-case rolling N-year annualised real returns for equity, bonds, and excess.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`data.load_shiller` or :func:`data.synthetic_world`.
    horizon : int
        Holding period in years (must be in HORIZONS).

    Returns a dict with:
      - ``n_windows``  — number of complete non-NaN rolling windows
      - ``eq_min``     — worst-case annualised real equity return (fraction)
      - ``eq_pct_neg`` — % of windows with negative real equity return
      - ``bd_min``     — worst-case annualised real bond return
      - ``excess_min`` — worst-case equity-minus-bond excess
      - ``excess_pct_neg`` — % of windows with equity beating bonds
      - ``eq_mean``, ``bd_mean``, ``excess_mean`` — mean across windows
      - ``tstat_excess`` — Newey-West HAC t-stat on per-window excess return
        (note: overlapping windows inflate effective N; treat with caution —
         we flag this honestly rather than using it naively)
    """
    key_eq = f"roll_{horizon}y_eq"
    key_bd = f"roll_{horizon}y_bd"
    key_ex = f"roll_{horizon}y_excess"
    eq = df[key_eq].dropna().to_numpy(dtype=float)
    bd = df[key_bd].dropna().to_numpy(dtype=float)
    ex = df[key_ex].dropna().to_numpy(dtype=float)

    n = min(len(eq), len(bd), len(ex))
    eq, bd, ex = eq[:n], bd[:n], ex[:n]

    out: dict = {
        "horizon_years": int(horizon),
        "n_windows": int(n),
        "eq_min": float(np.nanmin(eq)) if n else float("nan"),
        "eq_mean": float(np.nanmean(eq)) if n else float("nan"),
        "eq_pct_neg": float((eq < 0).mean() * 100) if n else float("nan"),
        "bd_min": float(np.nanmin(bd)) if n else float("nan"),
        "bd_mean": float(np.nanmean(bd)) if n else float("nan"),
        "excess_min": float(np.nanmin(ex)) if n else float("nan"),
        "excess_mean": float(np.nanmean(ex)) if n else float("nan"),
        "excess_pct_neg": float((ex < 0).mean() * 100) if n else float("nan"),
        "tstat_excess": float("nan"),
    }

    # HAC t-stat on excess — honest disclaimer: overlapping monthly windows means
    # effective sample size is much smaller than n; we compute it but flag the caveat.
    if n > 5:
        mu = ex.mean()
        e = ex - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat_excess"] = float(mu / se) if se > 0 else float("nan")

    return out


def all_horizon_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute :func:`worst_case_windows` for every horizon and return a tidy DataFrame."""
    rows = [worst_case_windows(df, h) for h in HORIZONS]
    return pd.DataFrame(rows).set_index("horizon_years")


# ---------------------------------------------------------------------------
# Null / control comparison
# ---------------------------------------------------------------------------
def shuffled_excess_control(
    df: pd.DataFrame, horizon: int, n_sim: int = 1000, seed: int = 151
) -> dict:
    """Bootstrap the distribution of worst-case excess returns under i.i.d. monthly returns.

    Shuffles the monthly equity and bond returns (breaking their time structure)
    and recomputes the rolling N-year excess.  This is the 'pure-luck' null:
    how often does a random-order world produce a positive excess on every window?
    Gives context for whether Siegel's track record is extraordinary or just
    the result of a long upward drift in equities.

    Returns:
      - ``p_always_positive`` — fraction of shuffled worlds where excess > 0 on all windows
      - ``worst_case_dist``   — array of worst-case excess returns across simulations
      - ``pct5_worst``        — 5th-percentile worst-case across simulations
    """
    rng = np.random.default_rng(seed)
    eq_ret = df["real_eq_ret"].dropna().to_numpy(dtype=float)
    bd_ret = df["real_bd_ret"].dropna().to_numpy(dtype=float)
    n = min(len(eq_ret), len(bd_ret))
    eq_ret, bd_ret = eq_ret[:n], bd_ret[:n]

    months = horizon * 12
    worst_cases = np.empty(n_sim)
    for i in range(n_sim):
        idx = rng.integers(0, n, n)
        eq_s = eq_ret[idx]
        bd_s = bd_ret[idx]
        tr_eq = np.cumprod(1.0 + eq_s)
        tr_bd = np.cumprod(1.0 + bd_s)
        # forward returns for each starting index t
        excesses = []
        for t in range(n - months):
            eq_ann = (tr_eq[t + months] / tr_eq[t]) ** (1.0 / horizon) - 1.0
            bd_ann = (tr_bd[t + months] / tr_bd[t]) ** (1.0 / horizon) - 1.0
            excesses.append(eq_ann - bd_ann)
        worst_cases[i] = min(excesses) if excesses else float("nan")

    valid = worst_cases[np.isfinite(worst_cases)]
    return {
        "horizon_years": int(horizon),
        "n_sim": int(n_sim),
        "p_always_positive": float((valid > 0).mean()),
        "pct5_worst": float(np.percentile(valid, 5)) if len(valid) else float("nan"),
        "pct50_worst": float(np.percentile(valid, 50)) if len(valid) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Decade-by-decade breakdown (the honest "there ARE bad stretches" chart)
# ---------------------------------------------------------------------------
def decade_returns(df: pd.DataFrame) -> pd.DataFrame:
    """10-year non-overlapping windows: equity, bond, and excess real return.

    Uses the non-overlapping windows starting at 1880, 1890, …, 2010 (or the
    nearest start in the data).  Each row is one decade.
    """
    years = df.index.year.unique()
    starts = range(int(years.min()), int(years.max()) - 9, 10)
    rows = []
    for y in starts:
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{y + 10}-01-01")
        sub = df[(df.index >= start) & (df.index < end)]
        if len(sub) < 100:
            continue
        eq_ret = sub["real_eq_ret"].to_numpy(dtype=float)
        bd_ret = sub["real_bd_ret"].to_numpy(dtype=float)
        eq_ann = (np.cumprod(1.0 + eq_ret)[-1]) ** (1.0 / 10.0) - 1.0
        bd_ann = (np.cumprod(1.0 + bd_ret)[-1]) ** (1.0 / 10.0) - 1.0
        rows.append(
            {
                "decade": int(y),
                "eq_ann_real": eq_ann,
                "bd_ann_real": bd_ann,
                "excess": eq_ann - bd_ann,
            }
        )
    return pd.DataFrame(rows).set_index("decade")


# ---------------------------------------------------------------------------
# Convenience re-export of data loader
# ---------------------------------------------------------------------------
def load_real() -> pd.DataFrame:
    """Convenience wrapper: load Shiller panel and return the rolling-return frame."""
    return load_shiller()
