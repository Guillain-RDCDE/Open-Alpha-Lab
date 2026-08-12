"""Strategy + inference for Study 891 — Insurance Float Engine.

The quantities that decide whether "float is a structural edge" survives contact with the
tape, all on monthly TOTAL returns, both legs measured excess-of-cash (minus BIL):

* **Excess-vs-excess Sharpe race** — annualised Sharpe of (insurer − cash) vs (market − cash);
  the *advantage* is the number the folklore predicts to be positive.
* **HAC *t* on the return difference** — Newey-West (Bartlett) *t* of the monthly
  (insurer − market) return, the honest significance test on a serially-correlated diff.
* **CAPM decomposition** — insurer excess on market excess: does a positive **alpha** survive
  once you subtract beta·market? (delivery-of-alpha test.)
* **Two-factor decomposition** — insurer excess on [market excess, bank − market spread]: the
  decisive test. If the "float premium" is really just financial-sector beta, the market +
  bank factors soak it up and the alpha collapses to zero.
* **Bootstrap Sharpe CI** — a circular-block-bootstrap CI on the insurer excess Sharpe
  (``quantlab.stats.sharpe_ci_bootstrap``) — is it even distinguishable from the market's?
* **Drawdowns, calendar-year table, era cut** — the structural story should hold across eras,
  not just in a crisis where insurers happened to fall less.
* **One-month-lag rotation probe + costed isolation trade** — the tradability side: a
  point-in-time (shift-by-one) "own insurers when they've been beating the market" rotation,
  and a long-insurer / short-market dollar-neutral book charged one-way costs × turnover +
  borrow on the short leg.

Reuses ``quantlab.stats`` (annualized_sharpe, sharpe_ci_bootstrap) and mirrors
``quantlab.analytics.mean_tstat_hac`` for the HAC primitive. Deterministic; pure numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.stats import annualized_sharpe, sharpe_ci_bootstrap

MONTHS = 12
NW_LAGS = 6  # Newey-West lag window for monthly series (~1.5·T^(1/3) at T ≈ 200)


# --------------------------------------------------------------------------- #
# HAC / Newey-West primitives
# --------------------------------------------------------------------------- #
def nw_mean_t(x, lags: int = NW_LAGS) -> tuple[float, float]:
    """Mean of ``x`` and its Newey-West (Bartlett) HAC *t*-statistic vs zero."""
    v = np.asarray(x, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 8:
        return float("nan"), float("nan")
    e = v - v.mean()
    s = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(v.mean()), float(v.mean() / se) if se > 0 else float("nan")


def _nw_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS of ``y`` on design ``X`` with a Newey-West covariance; returns (beta, se, r2)."""
    n = len(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    G = X * resid[:, None]
    S = G.T @ G
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        Gk = G[k:].T @ G[:-k]
        S += w * (Gk + Gk.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    denom = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / denom if denom > 0 else float("nan")
    return beta, se, r2


# --------------------------------------------------------------------------- #
# Excess helpers
# --------------------------------------------------------------------------- #
def excess(ret: pd.DataFrame, col: str, cash: str = "BIL") -> pd.Series:
    """Excess-of-cash monthly return of ``col`` (minus the cash-leg column)."""
    return (ret[col] - ret[cash]).rename(f"{col}_ex")


# --------------------------------------------------------------------------- #
# The excess-vs-excess Sharpe race + HAC t on the difference
# --------------------------------------------------------------------------- #
def sharpe_race(ret: pd.DataFrame, ins: str, mkt: str = "SPY", cash: str = "BIL",
                lags: int = NW_LAGS) -> dict:
    """Annualised excess Sharpe of insurer vs market, their advantage, and the HAC diff *t*.

    Both legs are excess-of-cash (minus ``cash``). ``advantage`` = insurer Sharpe − market
    Sharpe (the folklore predicts > 0). ``diff`` is the raw monthly (insurer − market) return;
    its Newey-West *t* is the honest significance of the out/under-performance.
    """
    df = ret[[ins, mkt, cash]].dropna()
    ins_ex = df[ins] - df[cash]
    mkt_ex = df[mkt] - df[cash]
    diff = df[ins] - df[mkt]
    mean_diff, t_diff = nw_mean_t(diff.values, lags)
    s_ins = annualized_sharpe(ins_ex, periods_per_year=MONTHS)
    s_mkt = annualized_sharpe(mkt_ex, periods_per_year=MONTHS)
    return {
        "n": int(len(df)),
        "start": str(df.index.min().date()), "end": str(df.index.max().date()),
        "sharpe_ins": s_ins, "sharpe_mkt": s_mkt, "advantage": s_ins - s_mkt,
        "diff_ann_pct": mean_diff * MONTHS * 100.0, "t_diff": t_diff,
    }


# --------------------------------------------------------------------------- #
# CAPM + two-factor (market + bank-spread) decomposition
# --------------------------------------------------------------------------- #
def capm(ret: pd.DataFrame, ins: str, mkt: str = "SPY", cash: str = "BIL",
         lags: int = NW_LAGS) -> dict:
    """CAPM on monthly EXCESS returns: insurer_ex = α + β·market_ex, Newey-West errors.

    ``alpha_ann`` (with its NW *t*) is the delivery-of-alpha test: a structural edge over the
    market must show a positive, significant alpha *after* subtracting β·market.
    """
    df = ret[[ins, mkt, cash]].dropna()
    y = (df[ins] - df[cash]).to_numpy(dtype=float)
    x = (df[mkt] - df[cash]).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), x])
    beta, se, r2 = _nw_ols(y, X, lags)
    return {
        "alpha_ann_pct": float(beta[0] * MONTHS * 100.0),
        "t_alpha": float(beta[0] / se[0]) if se[0] > 0 else float("nan"),
        "beta": float(beta[1]), "t_beta_vs1": float((beta[1] - 1.0) / se[1]) if se[1] > 0 else float("nan"),
        "r2": r2, "n": len(y),
    }


def decompose_financials(ret: pd.DataFrame, ins: str, mkt: str = "SPY", bank: str = "KBE",
                         cash: str = "BIL", lags: int = NW_LAGS) -> dict:
    """The decisive test — insurer excess on [market excess, (bank − market) spread].

    Regression ``ins_ex = α + β·mkt_ex + s·(bank_ex − mkt_ex)``. The bank-minus-market spread
    is a clean **financial-sector** factor orthogonal-ish to the market. If the insurer's
    apparent out-performance is really just financial-sector beta, ``s`` loads positive and
    significant while ``α`` collapses toward zero. A surviving positive ``α`` would be the
    *float-specific* premium the folklore claims.
    """
    df = ret[[ins, mkt, bank, cash]].dropna()
    y = (df[ins] - df[cash]).to_numpy(dtype=float)
    mkt_ex = (df[mkt] - df[cash]).to_numpy(dtype=float)
    bank_sp = ((df[bank] - df[cash]) - (df[mkt] - df[cash])).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), mkt_ex, bank_sp])
    beta, se, r2 = _nw_ols(y, X, lags)
    return {
        "alpha_ann_pct": float(beta[0] * MONTHS * 100.0),
        "t_alpha": float(beta[0] / se[0]) if se[0] > 0 else float("nan"),
        "beta_mkt": float(beta[1]),
        "load_bank": float(beta[2]), "t_load_bank": float(beta[2] / se[2]) if se[2] > 0 else float("nan"),
        "r2": r2, "n": len(y),
    }


def insurer_vs_bank(ret: pd.DataFrame, ins: str, bank: str = "KBE", lags: int = NW_LAGS) -> dict:
    """Within-financials check: monthly (insurer − bank) return, mean and HAC *t*.

    This is a *different* claim from the headline (it does not beat the market) — it asks
    whether float-funded insurers ride a better risk-adjusted path than spread-levered banks.
    """
    df = ret[[ins, bank]].dropna()
    diff = df[ins] - df[bank]
    mean_diff, t_diff = nw_mean_t(diff.values, lags)
    return {"diff_ann_pct": mean_diff * MONTHS * 100.0, "t_diff": t_diff, "n": int(len(df))}


# --------------------------------------------------------------------------- #
# Bootstrap Sharpe CI
# --------------------------------------------------------------------------- #
def bootstrap_sharpe(ret: pd.DataFrame, col: str, cash: str = "BIL",
                     n_boot: int = 2000, seed: int = 891) -> dict:
    """Circular-block-bootstrap 95% CI on the excess (minus cash) annualised Sharpe."""
    df = ret[[col, cash]].dropna()
    ex = df[col] - df[cash]
    r = sharpe_ci_bootstrap(ex, n_boot=n_boot, periods_per_year=MONTHS, seed=seed)
    return {"sharpe": r["sharpe"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "frac_negative": r["frac_negative"], "n": r["n_obs"]}


# --------------------------------------------------------------------------- #
# Performance / risk descriptives
# --------------------------------------------------------------------------- #
def ann_stats(ret: pd.DataFrame, col: str, cash: str = "BIL") -> dict:
    """CAGR, annualised vol, excess Sharpe, max drawdown on monthly TOTAL returns."""
    df = ret[[col, cash]].dropna()
    r = df[col]
    n = len(r)
    wealth = float((1.0 + r).prod())
    cagr = wealth ** (MONTHS / n) - 1.0 if n > 0 else float("nan")
    vol = float(r.std() * np.sqrt(MONTHS))
    ex = df[col] - df[cash]
    sharpe = annualized_sharpe(ex, periods_per_year=MONTHS)
    curve = (1.0 + r).cumprod()
    maxdd = float((curve / curve.cummax() - 1.0).min())
    return {"cagr_pct": cagr * 100.0, "vol_pct": vol * 100.0, "sharpe": sharpe,
            "maxdd_pct": maxdd * 100.0, "n": n}


def calendar_year_table(ret: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Calendar-year compounded TOTAL returns (%) per column (partial edge years included)."""
    yr = (1.0 + ret[cols]).groupby(ret.index.year).apply(lambda g: g.prod() - 1.0)
    return (yr * 100.0).round(2)


def era_table(ret: pd.DataFrame, eras: list[tuple[str, str, str]], ins: str,
              mkt: str = "SPY", bank: str = "KBE", cash: str = "BIL") -> pd.DataFrame:
    """Excess Sharpe of insurer / market / bank + the insurer−market advantage, per era."""
    rows = {}
    for lo, hi, tag in eras:
        sub = ret[(ret.index >= pd.Timestamp(lo)) & (ret.index <= pd.Timestamp(hi))]
        if len(sub) < 6:
            continue
        c = sub[cash]
        s_ins = annualized_sharpe(sub[ins] - c, periods_per_year=MONTHS)
        s_mkt = annualized_sharpe(sub[mkt] - c, periods_per_year=MONTHS)
        s_bank = annualized_sharpe(sub[bank] - c, periods_per_year=MONTHS)
        rows[tag] = {"n": int(len(sub)), "sharpe_ins": s_ins, "sharpe_mkt": s_mkt,
                     "sharpe_bank": s_bank, "advantage": s_ins - s_mkt}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Tradability — one-month-lag rotation + costed isolation trade
# --------------------------------------------------------------------------- #
def rotation_signal(ret: pd.DataFrame, ins: str, mkt: str = "SPY", lookback: int = 12) -> pd.Series:
    """Point-in-time rotation signal: hold the insurer when its trailing ``lookback``-month
    return has beaten the market, else hold the market.

    The trailing (insurer − market) cumulative return is computed through month ``t`` and
    then **shifted by one month**, so the position held over month ``t`` uses only information
    known at the close of ``t-1``: exactly one month of execution lag, no look-ahead.
    ``True`` = hold insurer, ``False`` = hold market.
    """
    trail_ins = (1.0 + ret[ins]).rolling(lookback).apply(np.prod, raw=True)
    trail_mkt = (1.0 + ret[mkt]).rolling(lookback).apply(np.prod, raw=True)
    sig = (trail_ins > trail_mkt).shift(1)
    return sig.fillna(False)


def rotation_strategy(ret: pd.DataFrame, ins: str, mkt: str = "SPY", cash: str = "BIL",
                      lookback: int = 12, cost_bps: float = 5.0, lags: int = NW_LAGS) -> dict:
    """Costed one-month-lag rotation: own the insurer when it's been beating the market, else
    the market. A full switch pays 2 × ``cost_bps`` one-way (sell one wrapper, buy the other).
    Reports the rotation's excess Sharpe vs always-market and its net annual return.
    """
    df = ret[[ins, mkt, cash]].dropna()
    sig = rotation_signal(df, ins, mkt, lookback)
    held = np.where(sig.values, df[ins].values, df[mkt].values)
    switches = int((sig != sig.shift(1)).iloc[1:].sum())
    n = len(df)
    cost_total = switches * 2.0 * cost_bps / 1e4
    gross = pd.Series(held, index=df.index)
    net = gross.copy()
    # spread the switching cost across the sample as an annualised drag on the mean
    net = net - cost_total / n
    ex_net = net - df[cash]
    return {
        "n": n, "switches": switches, "share_ins": float(sig.mean()),
        "sharpe_net": annualized_sharpe(ex_net, periods_per_year=MONTHS),
        "net_ann_pct": float(net.mean() * MONTHS * 100.0),
        "always_mkt_ann_pct": float(df[mkt].mean() * MONTHS * 100.0),
        "always_ins_ann_pct": float(df[ins].mean() * MONTHS * 100.0),
    }


def isolation_trade(ret: pd.DataFrame, ins: str, mkt: str = "SPY", cash: str = "BIL",
                    cost_bps_oneway: float = 5.0, borrow_annual_bps: float = 40.0,
                    turnover_per_year: float = 1.0, lags: int = NW_LAGS) -> dict:
    """Isolate the claimed edge: long insurer / short market (dollar-neutral, per $ of long).

    Gross monthly P&L = insurer − market. Charges: borrow on the short market leg
    (``borrow_annual_bps``/12) and one-way costs × NAV on ~``turnover_per_year`` rebalances of
    BOTH legs per year. Reports the net annual spread and its Newey-West *t* — the honest
    "can you actually pocket the float premium over the market?" number.
    """
    df = ret[[ins, mkt, cash]].dropna()
    gross = df[ins] - df[mkt]
    charge_m = borrow_annual_bps / 1e4 / 12.0 + 2.0 * cost_bps_oneway / 1e4 * turnover_per_year / 12.0
    net = gross - charge_m
    g_ann, g_t = nw_mean_t(gross.values, lags)
    n_ann, n_t = nw_mean_t(net.values, lags)
    return {
        "n": int(len(df)),
        "gross_ann_pct": g_ann * MONTHS * 100.0, "t_gross": g_t,
        "net_ann_pct": n_ann * MONTHS * 100.0, "t_net": n_t,
        "charge_ann_pct": charge_m * MONTHS * 100.0,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, ins: str = "KIE", mkt: str = "SPY",
                     bank: str = "KBE", cash: str = "BIL") -> dict:
    """Run the headline race + CAPM + two-factor on a synthetic world."""
    race = sharpe_race(world, ins, mkt, cash)
    cap = capm(world, ins, mkt, cash)
    two = decompose_financials(world, ins, mkt, bank, cash)
    return {
        "advantage": race["advantage"], "t_diff": race["t_diff"],
        "capm_alpha_ann_pct": cap["alpha_ann_pct"], "capm_t_alpha": cap["t_alpha"],
        "two_alpha_ann_pct": two["alpha_ann_pct"], "two_t_alpha": two["t_alpha"],
        "load_bank": two["load_bank"], "t_load_bank": two["t_load_bank"],
    }
