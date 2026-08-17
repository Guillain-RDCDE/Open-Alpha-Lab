"""Strategy + inference for Study 942 — The Inverse Tax.

The claim under test, stated at full strength: *an inverse ETF is a structurally worse way
to be short than simply being short — the expense ratio, the daily-reset path drag and the
financing all leak away, so anyone who can borrow the shares should short the index
directly.* This module builds the honest alternative and races it.

**The two arms** (per $1 of NAV, held continuously, no timing signal anywhere):

1. ``fund`` — the inverse ETF's own daily total return (SH, PSQ or SDS).
2. ``direct`` — a **directly-short book** rebalanced daily to the same constant −k× NAV
   exposure, priced honestly:

   ``r_direct = k * r_index_total + credit * |k| * rf - borrow - rebalance cost``

   * ``r_index_total`` is the underlying's **total** return — a share-short **owes the
     dividends**, which is why the total-return leg (not the price leg) is the right one.
   * ``rf`` is the ^IRX bill accrual, act/360 over calendar days, at the *previous*
     close's level.
   * ``credit`` is a **PROXY / ASSUMPTION**, not tape: how many units of the bill rate the
     shorter actually collects on a $1 NAV / $1 short. ``credit=0`` = a broker that pays
     nothing on either balance; ``credit=1`` (the base case) = interest on your own NAV
     cash but **no rebate on the short proceeds**, which is the ordinary retail
     arrangement; ``credit=2`` = the prime-brokerage case where the proceeds are rebated
     too. The whole verdict turns on this number, so it is swept.
   * ``borrow_bps`` is a **PROXY / ASSUMPTION**: the annual stock-loan fee on the
     underlying, act/360, charged on the |k|× notional. SPY and QQQ are general
     collateral; the base case is 30 bps and the sweep runs 0-100.
   * ``cost_bps`` is a one-way transaction cost **× NAV** charged on the daily
     re-leveraging turnover of the direct book (the fund's own turnover is already inside
     its NAV, so charging it again would double-count). The turnover a constant −k× book
     actually has to trade after a move of ``r`` is ``|k(k-1)| × |r|`` of NAV — 2×|r| at
     −1×, 6×|r| at −2× — not ``|k| × |r|``.

**The two arms are not the same mandate.** SH/PSQ/SDS track the **price** index, so their
holder never owes the distributions; a share-short owes them. That dividend term is the
single largest piece of the raw gap, and it is a *mandate* difference as much as an
efficiency one. ``same_mandate_gap`` re-runs the race with the fund charged the dividends
too — the strictly like-for-like exposure comparison — and it is the honest counterweight
to the headline. Both numbers are reported; they carry opposite signs.

**One execution lag, once:** the exposure set at the close of day ``t-1`` earns day ``t``'s
return, and the rate financed at over day ``t`` is the ``^IRX`` level published at the
close of ``t-1``. There is no forecasting signal in this study — both arms hold a constant
exposure — so this rebalance/financing convention is the only lag, and it is applied
identically to both arms.

**The headline** is the daily gap ``r_fund - r_direct``, its annualised mean, its HAC
(Newey-West) *t*, and a circular block-bootstrap CI. Around it: an accounting
decomposition (dividends not owed / financing passthrough / expense ratio / residual), a
path-drag table across holding horizons, an era cut, a rate-regime cut, and the two proxy
sweeps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk kit)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0; automatic lag if None."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu, std = r.mean(), r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and wealth.iloc[-1] > 0 else float("nan"))
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def annualised_pct(daily: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised arithmetic mean of a daily series, in percent."""
    return float(pd.Series(daily).astype(float).dropna().mean() * periods_per_year * 100.0)


# --------------------------------------------------------------------------- #
# The two arms
# --------------------------------------------------------------------------- #
def prepare(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    rate: str = "IRX",
    cash: str = "BIL",
) -> pd.DataFrame:
    """Align the tape into daily return legs for one fund/underlying pair.

    Columns returned: ``r_fund`` (the ETF's total return), ``r_index_tr`` (the
    underlying's **total** return — what a share-short owes), ``r_index_px`` (the
    underlying's **price-only** return — what the fund's benchmark tracks), ``rf`` (the
    ^IRX act/360 accrual at the previous close's level) and ``r_cash`` (BIL's realised
    total return, the excess-of-cash yardstick).
    """
    from . import data as _data

    cols = [fund, index, index + "_px", rate, cash]
    d = px[cols].dropna().sort_index()
    out = pd.DataFrame(
        {
            "r_fund": d[fund].pct_change(),
            "r_index_tr": d[index].pct_change(),
            "r_index_px": d[index + "_px"].pct_change(),
            "rf": _data.cash_rate_daily(d[rate]),
            "r_cash": d[cash].pct_change(),
        }
    )
    out["days"] = pd.Series(d.index, index=d.index).diff().dt.days.astype(float)
    return out.dropna()


def direct_short(
    legs: pd.DataFrame,
    k: float = -1.0,
    credit: float = 1.0,
    borrow_bps: float = 30.0,
    cost_bps: float = 1.0,
) -> pd.Series:
    """Daily return of the honest directly-short book at constant −k× NAV exposure.

    ``credit`` and ``borrow_bps`` are PROXY inputs (see the module docstring); both are
    swept in ``credit_borrow_sweep``. The rebalance cost is one-way × NAV on the turnover
    the daily re-leveraging actually requires. After a move of ``r`` the book's NAV is
    ``1 + k·r`` while its notional is ``k(1 + r)``, so restoring the −k× target takes
    ``|k(k-1)| × |r|`` of NAV: **2×|r| at −1×, 6×|r| at −2×** (not ``|k| × |r|``, which
    understates it and would flatter the direct book).
    """
    borrow = (borrow_bps * 1e-4) * legs["days"] / 360.0 * abs(k)
    turnover = abs(k * (k - 1.0)) * legs["r_index_tr"].abs()
    tc = turnover * cost_bps * 1e-4
    return (k * legs["r_index_tr"] + credit * abs(k) * legs["rf"] - borrow - tc).rename("r_direct")


def gap_series(
    legs: pd.DataFrame,
    k: float = -1.0,
    credit: float = 1.0,
    borrow_bps: float = 30.0,
    cost_bps: float = 1.0,
) -> pd.Series:
    """The daily gap: inverse fund minus the honest direct short. Positive = fund wins."""
    return (legs["r_fund"] - direct_short(legs, k, credit, borrow_bps, cost_bps)).rename("gap")


def same_mandate_gap(
    legs: pd.DataFrame,
    k: float = -1.0,
    credit: float = 1.0,
    borrow_bps: float = 30.0,
    cost_bps: float = 1.0,
) -> pd.Series:
    """The gap with the **fund charged the index's dividends too** — same-mandate race.

    The headline race is *product against product*: buy the fund, or short the shares. On
    that comparison the fund's benchmark is the **price** index and the shorter's is the
    **total-return** stream, so the distributions are part of what you are choosing
    between. But they are also a difference of mandate, not only of efficiency — the two
    arms are not short the same payoff.

    This function removes that difference by debiting the fund ``|k| ×`` the realised
    distribution yield, putting both arms on the total-return leg. It is the strictly
    like-for-like exposure comparison, and on the real tape it carries the **opposite
    sign** to the headline. Reporting both is the point of the study.
    """
    div = abs(k) * (legs["r_index_tr"] - legs["r_index_px"])
    return (gap_series(legs, k, credit, borrow_bps, cost_bps) - div).rename("gap_same_mandate")


def hac_lag_profile(x, lag_grid=(0, 1, 2, 5, 9, 21, 63, 126, 252)) -> list[dict]:
    """HAC *t* of a series across a grid of Bartlett truncation lags.

    The automatic bandwidth is one choice among many, and the daily gap carries a large
    *negative* lag-1 autocorrelation (ETF close vs index close noise), which makes the HAC
    variance materially smaller than the i.i.d. one. Quoting a single *t* would hide that,
    so the whole profile is reported: ``lags=0`` is the plain i.i.d. *t*.
    """
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    rows = [{"lags": int(L), "t": newey_west_t(arr, lags=int(L))} for L in lag_grid]
    rows.append({"lags": None, "t": newey_west_t(arr)})
    return rows


# --------------------------------------------------------------------------- #
# The headline race
# --------------------------------------------------------------------------- #
def race(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    k: float = -1.0,
    credit: float = 1.0,
    borrow_bps: float = 30.0,
    cost_bps: float = 1.0,
) -> dict:
    """Race the inverse fund against the honest direct short on the real (or synthetic) tape.

    Both arms are also reported **excess-of-cash** (minus BIL's realised total return) so
    the Sharpe comparison is excess-vs-excess. Note that both arms are *short* arms in a
    market that rose over the sample, so both excess Sharpes are deeply negative; the
    quantity that carries the verdict is the **gap**, not either Sharpe on its own.
    """
    legs = prepare(px, fund=fund, index=index)
    r_dir = direct_short(legs, k, credit, borrow_bps, cost_bps)
    gap = (legs["r_fund"] - r_dir).rename("gap")
    gap_sm = same_mandate_gap(legs, k, credit, borrow_bps, cost_bps)

    # Excess-of-cash on BOTH arms — the Sharpe comparison is excess-vs-excess.
    e_fund = (legs["r_fund"] - legs["r_cash"]).rename("e_fund")
    e_dir = (r_dir - legs["r_cash"]).rename("e_direct")
    s_fund, s_dir = summary(e_fund), summary(e_dir)

    return {
        "fund": fund, "index": index, "k": k,
        "credit": credit, "borrow_bps": borrow_bps, "cost_bps": cost_bps,
        "n_days": int(len(gap)),
        "start": legs.index[0], "end": legs.index[-1],
        "gap_ann_pct": annualised_pct(gap),
        "gap_t": newey_west_t(gap.to_numpy()),
        "gap_t_iid": newey_west_t(gap.to_numpy(), lags=0),
        "gap_sd_bps": float(gap.std(ddof=1) * 1e4),
        # Same-mandate counterweight: the fund charged the dividends too.
        "gap_same_mandate_ann_pct": annualised_pct(gap_sm),
        "gap_same_mandate_t": newey_west_t(gap_sm.to_numpy()),
        "gap_same_mandate": gap_sm,
        "s_fund": s_fund,
        "s_direct": s_dir,
        "sharpe_diff": s_fund["sharpe"] - s_dir["sharpe"],
        "rf_ann_pct": annualised_pct(legs["rf"]),
        "legs": legs, "gap": gap, "r_direct": r_dir,
        "e_fund": e_fund, "e_direct": e_dir,
    }


# --------------------------------------------------------------------------- #
# The accounting decomposition
# --------------------------------------------------------------------------- #
def decompose(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    k: float = -1.0,
    expense_ratio_pct: float = 0.89,
) -> dict:
    """Split the *raw* structural gap into its four economic pieces.

    The raw gap is taken at ``credit=0, borrow_bps=0, cost_bps=0`` — i.e. against a
    directly-short book that collects **no** interest at all — so every financing effect
    shows up as a term rather than being netted away silently:

    ``raw_gap = dividends_not_owed + financing_passthrough - expense_ratio + residual``

    * ``dividends_not_owed`` = ``|k| ×`` the underlying's realised distribution yield
      (total-return minus price-only return). A share-short pays it; the fund's benchmark
      is the **price** index, so the fund's holder does not.
    * ``financing_passthrough`` = ``gamma × |k| ×`` the annualised bill rate, where
      ``gamma`` is the OLS coefficient on ``|k| × rf`` in the regression of the fund's
      daily return on its exposure leg and the rate leg. ``gamma`` is measured, not
      assumed.
    * ``expense_ratio`` is a **PROXY / ASSUMPTION** — the prospectus gross figure, a
      stated non-tape input.
    * ``residual`` is whatever is left: swap spread, reset slippage, tracking error.

    Also returns the regression's ``beta`` (realised exposure vs the −k× target) and
    ``alpha`` (its intercept, annualised).

    **Spec sensitivity, stated.** ``gamma`` depends on which exposure leg the regression
    uses. The headline spec runs the fund on the **total-return** leg (matching the
    replicate it is raced against); ``gamma_px`` re-estimates it on the **price** leg (what
    the funds actually track), and the two differ by up to ~0.2 rate units. The raw gap
    itself is spec-free — only the *split* between the financing term and the residual
    moves, so both are quoted.
    """
    legs = prepare(px, fund=fund, index=index)
    raw = gap_series(legs, k=k, credit=0.0, borrow_bps=0.0, cost_bps=0.0)

    y = legs["r_fund"].to_numpy()
    rate_leg = (abs(k) * legs["rf"]).to_numpy()

    def _fit(exposure: np.ndarray) -> tuple[float, float, float]:
        X = np.column_stack([np.ones(len(legs)), exposure, rate_leg])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(coef[0]), float(coef[1]), float(coef[2])

    alpha, beta, gamma = _fit((k * legs["r_index_tr"]).to_numpy())
    _, _, gamma_px = _fit((k * legs["r_index_px"]).to_numpy())

    div_ann_pct = annualised_pct(legs["r_index_tr"] - legs["r_index_px"])
    rf_ann_pct = annualised_pct(legs["rf"])

    div_term = abs(k) * div_ann_pct
    fin_term = gamma * abs(k) * rf_ann_pct
    raw_pct = annualised_pct(raw)
    residual = raw_pct - div_term - fin_term + expense_ratio_pct

    return {
        "fund": fund, "k": k, "n_days": int(len(legs)),
        "raw_gap_ann_pct": raw_pct,
        "div_yield_ann_pct": div_ann_pct,
        "rf_ann_pct": rf_ann_pct,
        "dividends_not_owed_pct": div_term,
        "financing_passthrough_pct": fin_term,
        "expense_ratio_pct": expense_ratio_pct,
        "residual_pct": residual,
        "beta": beta, "gamma": gamma, "gamma_px": gamma_px,
        "financing_passthrough_px_pct": gamma_px * abs(k) * rf_ann_pct,
        "alpha_ann_pct": alpha * TRADING_DAYS * 100.0,
    }


def breakeven_credit(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    k: float = -1.0,
    borrow_bps: float = 30.0,
    cost_bps: float = 1.0,
) -> float:
    """The rebate credit ``c`` at which the gap is exactly zero.

    The gap is linear in ``c`` (each unit of credit costs the direct book ``|k| × rf``),
    so two evaluations pin it down exactly. Below the break-even the fund wins; above it
    the direct short wins. This single number is the study's honest answer: *it depends on
    what your broker pays you.*
    """
    legs = prepare(px, fund=fund, index=index)
    g0 = annualised_pct(gap_series(legs, k, 0.0, borrow_bps, cost_bps))
    g1 = annualised_pct(gap_series(legs, k, 1.0, borrow_bps, cost_bps))
    slope = g1 - g0
    if slope == 0:
        return float("nan")
    return float(-g0 / slope)


# --------------------------------------------------------------------------- #
# Bootstrap, era cut, rate-regime cut, sweeps
# --------------------------------------------------------------------------- #
def bootstrap_gap_ci(
    gap: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 942,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the **annualised mean** of the daily gap (percent).

    Blocks of ``block`` consecutive days preserve the volatility clustering and the
    autocorrelation the daily tracking error carries.
    """
    r = np.asarray(pd.Series(gap).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    point = float(r.mean() * TRADING_DAYS * 100.0)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean() * TRADING_DAYS * 100.0
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_obs": int(n),
            "n_boot": int(n_boot), "block": int(block)}


def era_cut(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    k: float = -1.0,
    split: str = "2016-01-01",
    credit: float = 1.0,
    borrow_bps: float = 30.0,
    cost_bps: float = 1.0,
) -> dict:
    """Split the sample at ``split`` and re-measure the gap on each half."""
    legs = prepare(px, fund=fund, index=index)
    gap = gap_series(legs, k, credit, borrow_bps, cost_bps)
    out = {}
    for tag, mask in [("early", gap.index < pd.Timestamp(split)),
                      ("late", gap.index >= pd.Timestamp(split))]:
        g = gap[mask]
        out[tag] = None if len(g) < 60 else {
            "n_days": int(len(g)),
            "gap_ann_pct": annualised_pct(g),
            "gap_t": newey_west_t(g.to_numpy()),
            "rf_ann_pct": annualised_pct(legs["rf"][mask]),
        }
    return out


def rate_regime_cut(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    k: float = -1.0,
    threshold_pct: float = 1.0,
    credit: float = 1.0,
    borrow_bps: float = 30.0,
    cost_bps: float = 1.0,
) -> dict:
    """Cut the sample by the *prevailing bill rate*, the variable the gap should hinge on.

    The conditioning variable (the previous close's ^IRX level) is known at the start of
    each day, so this is a legitimate ex-ante partition, not a look-ahead sort.
    """
    legs = prepare(px, fund=fund, index=index)
    gap = gap_series(legs, k, credit, borrow_bps, cost_bps)
    rf_level_pct = legs["rf"] / legs["days"] * 360.0 * 100.0
    out = {}
    for tag, mask in [("zirp", rf_level_pct < threshold_pct),
                      ("normal", rf_level_pct >= threshold_pct)]:
        g = gap[mask]
        out[tag] = None if len(g) < 60 else {
            "n_days": int(len(g)),
            "gap_ann_pct": annualised_pct(g),
            "gap_t": newey_west_t(g.to_numpy()),
            "rf_ann_pct": annualised_pct(legs["rf"][mask]),
        }
    out["threshold_pct"] = threshold_pct
    return out


def credit_borrow_sweep(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    k: float = -1.0,
    credits=(0.0, 0.5, 1.0, 1.5, 2.0),
    borrows=(0.0, 30.0, 60.0, 100.0),
    cost_bps: float = 1.0,
) -> list[dict]:
    """Sweep the two PROXY inputs jointly: the rebate credit and the stock-loan fee."""
    legs = prepare(px, fund=fund, index=index)
    rows = []
    for c in credits:
        for b in borrows:
            g = gap_series(legs, k, c, b, cost_bps)
            rows.append({"credit": c, "borrow_bps": b,
                         "gap_ann_pct": annualised_pct(g),
                         "gap_t": newey_west_t(g.to_numpy())})
    return rows


def cost_sweep(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    k: float = -1.0,
    credit: float = 1.0,
    borrow_bps: float = 30.0,
    cost_grid=(0.0, 1.0, 5.0, 10.0),
) -> list[dict]:
    """Sweep the direct book's one-way rebalance cost (× NAV) from gross to punitive."""
    legs = prepare(px, fund=fund, index=index)
    rows = []
    for cb in cost_grid:
        g = gap_series(legs, k, credit, borrow_bps, cb)
        rows.append({"cost_bps": cb, "gap_ann_pct": annualised_pct(g),
                     "gap_t": newey_west_t(g.to_numpy())})
    return rows


# --------------------------------------------------------------------------- #
# Path drag across holding horizons
# --------------------------------------------------------------------------- #
def path_drag(
    px: pd.DataFrame,
    fund: str = "SH",
    index: str = "SPY",
    k: float = -1.0,
    horizons=(21, 63, 252),
    credit: float = 1.0,
    borrow_bps: float = 30.0,
    cost_bps: float = 1.0,
) -> list[dict]:
    """Compare three books over non-overlapping holding windows of each horizon.

    * ``fund`` — the ETF, compounded over the window.
    * ``reset`` — the honest direct short **rebalanced daily**, compounded.
    * ``static`` — a short put on once and left alone: ``k ×`` the window's total return
      plus the financing accrued, with no re-leveraging at all.

    ``reset - static`` is the **pure daily-reset path drag** — and it belongs to the
    *constant-leverage discipline*, not to the ETF wrapper: a self-managed book that keeps
    its exposure at −k× pays exactly the same drag. ``fund - reset`` is what the wrapper
    itself costs or saves.
    """
    legs = prepare(px, fund=fund, index=index)
    r_dir = direct_short(legs, k, credit, borrow_bps, cost_bps)
    rows = []
    for H in horizons:
        n = len(legs)
        f_c, s_c, d_c = [], [], []
        for i in range(0, n - H + 1, H):
            w = legs.iloc[i:i + H]
            wd = r_dir.iloc[i:i + H]
            idx_c = float((1.0 + w["r_index_tr"]).prod() - 1.0)
            rf_c = float(w["rf"].sum())
            bor_c = (borrow_bps * 1e-4) * float(w["days"].sum()) / 360.0 * abs(k)
            f_c.append(float((1.0 + w["r_fund"]).prod() - 1.0))
            d_c.append(float((1.0 + wd).prod() - 1.0))
            s_c.append(k * idx_c + credit * abs(k) * rf_c - bor_c)
        f_a, d_a, s_a = np.array(f_c), np.array(d_c), np.array(s_c)
        rows.append({
            "horizon_days": H, "n_windows": int(len(f_a)),
            "reset_minus_static_mean_pp": float((d_a - s_a).mean() * 100.0),
            "reset_minus_static_median_pp": float(np.median(d_a - s_a) * 100.0),
            "fund_minus_reset_mean_pp": float((f_a - d_a).mean() * 100.0),
            "fund_minus_reset_median_pp": float(np.median(f_a - d_a) * 100.0),
            "fund_minus_static_mean_pp": float((f_a - s_a).mean() * 100.0),
            "fund_minus_static_median_pp": float(np.median(f_a - s_a) * 100.0),
        })
    return rows


def financing_crosscheck(px: pd.DataFrame) -> dict:
    """Is the ^IRX construction fair to BIL, the tradable cash leg?

    BIL is the instrument an investor can actually hold; ^IRX is the rate the model
    accrues. If the two annualise within a few tens of bps over the common window, the
    financing model is not quietly flattering either arm.
    """
    from . import data as _data

    d = px[["IRX", "BIL"]].dropna()
    rf = _data.cash_rate_daily(d["IRX"]).dropna()
    r_bil = d["BIL"].pct_change().reindex(rf.index).dropna()
    rf = rf.reindex(r_bil.index)
    return {
        "n_days": int(len(rf)),
        "irx_ann_pct": annualised_pct(rf),
        "bil_ann_pct": annualised_pct(r_bil),
        "diff_ann_pct": annualised_pct(rf) - annualised_pct(r_bil),
    }


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Re-label a synthetic tape into the column names ``prepare`` expects."""
    return pd.DataFrame({
        "FUND": prices["fund"],
        "IDX": prices["index_tr"],
        "IDX_px": prices["index_px"],
        "IRX": prices["rate_pct"],
        "BIL": prices["cash"],
    })


def synthetic_detect(prices: pd.DataFrame, k: float = -1.0, credit: float = 1.0) -> dict:
    """Measure the gap on a synthetic tape with a known planted tax.

    At ``signal_strength=1`` the measured annualised gap must land near
    ``-truth['tax_ann']``; at ``signal_strength=0`` it must sit at zero. Proves the
    harness recovers a planted effect and stays quiet on the null — it never supports a
    real-tape stamp.
    """
    frame = synthetic_frame(prices)
    r = race(frame, fund="FUND", index="IDX", k=k, credit=credit,
             borrow_bps=0.0, cost_bps=0.0)
    return {
        "gap_ann_pct": r["gap_ann_pct"],
        "gap_t": r["gap_t"],
        "n_days": r["n_days"],
        "rf_ann_pct": r["rf_ann_pct"],
    }
