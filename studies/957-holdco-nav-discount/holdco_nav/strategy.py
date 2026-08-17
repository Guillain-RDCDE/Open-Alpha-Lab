"""Strategy + inference for Study 957 — Holdco Discount.

Two questions, in order.

**(a) Does the discount mean-revert?** For each holdco we build the observable-NAV proxy
from **price-only** closes, ``nav_t = k * P_stake_t + other_per_share``, and the discount
``d_t = 1 - P_holdco_t / nav_t``. Because the *level* of ``d`` inherits every error in the
``k`` / ``other_per_share`` assumptions, the tests use the **trailing z-score**
``z_t = (d_t - mean_{t-504..t} d) / sd_{t-504..t} d`` — each name standardised against its
own preceding two years, using only data through ``t``. Mean reversion is then a
predictive regression of the forward change in the discount on today's z, with a
Newey-West *t*; ``halflife`` reports the AR(1) half-life of the raw series for scale.

**(b) Does buying the wide discount pay after costs?** The tradable expression is the
*hedged pair*: long the holdco and short the stake **one for one in money terms**. That
ratio is not a guess — since ``P_holdco = (1 - d) * k * P_stake``, taking logs gives
``log P_holdco = log(1 - d) + log(k * P_stake)``, so a dollar-neutral long/short earns
exactly the change in ``log(1 - d)`` and nothing else. (The tempting alternative, shorting
the whole look-through stake value ``k * P_stake / P_holdco``, is the right hedge only if
the gap is fixed in *money*; on a proportional gap it leaves a residual short. It is kept
as ``hedge="lookthrough"`` and swept.) Two arms:

1. **always_on** — hold the equal-weighted hedged pair across every available name, every
   day. The structural control: it earns whatever the discount drifts, with no timing.
2. **timed** — hold a name's hedged pair only when its discount is unusually **wide**
   (``z_t >= enter``), exit when it has normalised (``z_t <= exit_``). Equal weight across
   whichever names are active; fully in cash when none are.

One execution lag throughout: the z formed from closes through day ``t`` sets the position
held over day ``t+1``. Returns come from **total-return** closes (dividends are received on
the long leg and paid away on the short). Costs are one-way bps on the traded notional of
*both* legs; the short leg pays a **borrow** fee on its notional, and both are swept.

Because a dollar-neutral long/short pair is financed out of collateral that itself earns
cash, the pair's return **is already excess-of-cash** — so ``timed`` versus ``always_on``
is an excess-of-cash versus excess-of-cash race by construction. The separate long-only
race (``long_only_race``) subtracts an explicit cash leg to keep the same footing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

Z_WINDOW = 504          # two years of trailing history for the standardisation
Z_MIN_PERIODS = 252     # a name only becomes eligible after one year of history
ENTER_Z = 1.0           # buy when the discount is >= 1 trailing sd wider than usual
EXIT_Z = 0.0            # sell once it is back at its own trailing mean
COST_BPS = 10.0         # one-way, per leg, on traded notional
BORROW_BPS = 100.0      # annualised borrow on the short leg's notional
FWD_DAYS = 126          # the six-month horizon for the mean-reversion regression


# --------------------------------------------------------------------------- #
# Inference primitives (shared house implementations)
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def hac_ols(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """Univariate OLS ``y = a + b x`` with a Newey-West *t* on the slope.

    Used for the mean-reversion regression, where overlapping forward windows make the
    residuals strongly autocorrelated — an OLS *t* there would be pure fiction.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 30 or x.std(ddof=1) == 0:
        return {"n": int(n), "beta": float("nan"), "alpha": float("nan"),
                "t_beta": float("nan"), "r2": float("nan")}
    if lags is None:
        lags = max(int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), FWD_DAYS)
    xm, ym = x.mean(), y.mean()
    beta = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    alpha = float(ym - beta * xm)
    resid = y - alpha - beta * x
    sxx = float(((x - xm) ** 2).sum())
    # HAC meat: sum of weighted autocovariances of (x-xbar)*resid
    g = (x - xm) * resid
    s = float(g @ g)
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        s += 2.0 * w * float(g[l:] @ g[:-l])
    var_b = s / (sxx ** 2)
    t_beta = float(beta / np.sqrt(var_b)) if var_b > 0 else float("nan")
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")
    return {"n": int(n), "beta": beta, "alpha": alpha, "t_beta": t_beta, "r2": r2}


def hac_ols_panel(pairs: list[tuple[pd.Series, pd.Series]], lags: int) -> dict:
    """Pooled OLS ``y = a + b x`` across names, with a Driscoll-Kraay standard error.

    Two things would wreck a naive pooled *t* here and both are handled: the forward
    windows **overlap** in time (so residuals are autocorrelated out to ``h`` lags), and the
    names are **contemporaneously correlated** (two of ours literally share an underlying,
    Tencent). So the score contributions are first summed *across names within each calendar
    day* and only then run through a Bartlett kernel — i.e. Driscoll & Kraay (1998) — rather
    than concatenated end to end, which would splice one name's last day onto the next
    name's first and pretend seven correlated series were one long independent one.
    """
    xs = [p[1] for p in pairs]
    ys = [p[0] for p in pairs]
    if not xs:
        return {"n": 0, "beta": float("nan"), "t_beta": float("nan"), "r2": float("nan")}
    all_x = np.concatenate([np.asarray(s, dtype=float) for s in xs])
    all_y = np.concatenate([np.asarray(s, dtype=float) for s in ys])
    n = all_x.size
    if n < 60 or all_x.std(ddof=1) == 0:
        return {"n": int(n), "beta": float("nan"), "t_beta": float("nan"), "r2": float("nan")}
    xm, ym = all_x.mean(), all_y.mean()
    sxx = float(((all_x - xm) ** 2).sum())
    beta = float(((all_x - xm) * (all_y - ym)).sum() / sxx)
    alpha = float(ym - beta * xm)
    ss_tot = float(((all_y - ym) ** 2).sum())
    resid_all = all_y - alpha - beta * all_x
    r2 = float(1.0 - (resid_all @ resid_all) / ss_tot) if ss_tot > 0 else float("nan")

    # per-day score, summed across names
    scores = []
    for y_s, x_s in pairs:
        e = y_s - alpha - beta * x_s
        scores.append(((x_s - xm) * e).rename("g"))
    G = pd.concat(scores, axis=1).sum(axis=1, min_count=1).dropna().sort_index().to_numpy()
    m = G.size
    if m < 10:
        return {"n": int(n), "beta": beta, "t_beta": float("nan"), "r2": r2}
    s = float(G @ G)
    for l in range(1, min(lags, m - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        s += 2.0 * w * float(G[l:] @ G[:-l])
    if s <= 0:
        return {"n": int(n), "beta": beta, "t_beta": float("nan"), "r2": r2}
    t_beta = float(beta / np.sqrt(s / (sxx ** 2)))
    return {"n": int(n), "beta": beta, "alpha": alpha, "t_beta": t_beta, "r2": r2,
            "n_days": int(m)}


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass an excess series)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"n_days": int(n), "cagr": float("nan"), "sharpe": float("nan"),
                "vol_ann": float("nan"), "max_drawdown": float("nan"),
                "mean_daily_bps": float("nan"), "tstat": float("nan")}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan"),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def bootstrap_sharpe_ci(excess: pd.Series, n_boot: int = 2000, block: int = 21,
                        seed: int = 957, periods_per_year: int = TRADING_DAYS,
                        alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of an excess-return series."""
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    ann = np.sqrt(periods_per_year)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * ann) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        ssd = s.std(ddof=1)
        if ssd > 0:
            boots[b] = s.mean() / ssd * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


# --------------------------------------------------------------------------- #
# The discount series
# --------------------------------------------------------------------------- #
def discount(holdco_px: pd.Series, stake_px: pd.Series, k: float,
             other_per_share: float = 0.0) -> pd.Series:
    """Observable-NAV discount from **price-only** closes: ``1 - P_holdco / (k*P_stake + o)``.

    Positive = the holdco trades *below* the marked value of its stake. ``k`` and ``o`` are
    assumptions (see ``data.STAKES``); a constant error in either shifts and rescales this
    series but leaves its trailing-standardised version, which every test uses, nearly
    unchanged.
    """
    idx = holdco_px.index.intersection(stake_px.index)
    nav = k * stake_px.reindex(idx) + other_per_share
    nav = nav.where(nav > 0)
    return (1.0 - holdco_px.reindex(idx) / nav).rename("discount")


def trailing_z(d: pd.Series, window: int = Z_WINDOW,
               min_periods: int = Z_MIN_PERIODS) -> pd.Series:
    """Standardise the discount against its own *preceding* ``window`` days.

    Uses only information available at ``t`` (the rolling window ends at ``t``, inclusive),
    so no future observation ever enters the z. NaN through the warm-up.
    """
    mu = d.rolling(window, min_periods=min_periods).mean()
    sd = d.rolling(window, min_periods=min_periods).std(ddof=1)
    return ((d - mu) / sd.where(sd > 0)).rename("z")


def hedge_ratio(holdco_px: pd.Series, stake_px: pd.Series, k: float) -> pd.Series:
    """Look-through hedge: the whole stake notional the holdco owns, per unit held.

    ``h_t = k * P_stake_t / P_holdco_t = 1/(1-d_t)``. This is the correct hedge only under a
    *fixed money gap*; on a proportional discount it over-hedges and leaves the pair net
    short the stake. Retained for the ``hedge="lookthrough"`` robustness variant and for
    disclosure — the headline uses the dollar-neutral hedge below.
    """
    idx = holdco_px.index.intersection(stake_px.index)
    return (k * stake_px.reindex(idx) / holdco_px.reindex(idx)).rename("hedge_lt")


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
def build_panel(px_only: pd.DataFrame, px_total: pd.DataFrame, stakes: dict,
                safe=lambda s: s.replace("=", "").replace("^", "").replace("/", ""),
                k_mult: float = 1.0, other_mult: float = 1.0,
                names=None, z_window: int = Z_WINDOW,
                z_min_periods: int = Z_MIN_PERIODS) -> dict[str, pd.DataFrame]:
    """Assemble, per holdco, a frame of everything the backtest needs.

    Columns: ``d`` (discount, price-only), ``z`` (trailing standardised discount),
    ``h`` (look-through hedge ratio), ``r_holdco`` / ``r_stake`` (total-return daily simple
    returns). ``k_mult`` / ``other_mult`` scale the NAV assumptions for the calibration
    sweep; ``z_window`` is exposed so the standardisation length can be swept too (it is a
    free parameter, and a free parameter that is never varied is a hidden choice). Each name
    is sliced to its own ``start`` / ``end``.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, cfg in stakes.items():
        if names is not None and name not in names:
            continue
        hc, stc = safe(cfg["holdco"]), safe(cfg["stake"])
        if hc not in px_only.columns or stc not in px_only.columns:
            continue
        hp = px_only[hc].dropna()
        sp = px_only[stc].dropna()
        k = cfg["k"] * k_mult
        o = cfg["other_per_share"] * other_mult
        d = discount(hp, sp, k, o)
        h_lt = hedge_ratio(hp, sp, k)
        r_h = px_total[hc].dropna().pct_change().rename("r_holdco")
        r_s = px_total[stc].dropna().pct_change().rename("r_stake")
        z = trailing_z(d, window=z_window, min_periods=min(z_min_periods, z_window))
        frame = pd.concat([d, z, h_lt, r_h, r_s], axis=1, sort=False).sort_index()
        frame = frame.rename(columns={"discount": "d", "hedge_lt": "h_lt"})
        frame["h"] = 1.0                      # the headline dollar-neutral hedge
        frame = frame[frame.index >= pd.Timestamp(cfg["start"])]
        if cfg.get("end"):
            frame = frame[frame.index <= pd.Timestamp(cfg["end"])]
        out[name] = frame.dropna(subset=["d", "h_lt"])
    return out


def build_panel_synthetic(frames: dict[str, pd.DataFrame], k: float = 1.0) -> dict[str, pd.DataFrame]:
    """Same shape as ``build_panel`` but from ``data.synthetic_panel`` output.

    The synthetic tape pays no dividends, so price-only and total-return coincide there.
    """
    out = {}
    for name, f in frames.items():
        d = discount(f["holdco"], f["stake"], k, 0.0)
        h = hedge_ratio(f["holdco"], f["stake"], k)
        frame = pd.DataFrame({
            "d": d, "z": trailing_z(d), "h": 1.0, "h_lt": h,
            "r_holdco": f["holdco"].pct_change(),
            "r_stake": f["stake"].pct_change(),
        })
        out[name] = frame.dropna(subset=["d", "h_lt"])
    return out


# --------------------------------------------------------------------------- #
# (a) Does the discount mean-revert?
# --------------------------------------------------------------------------- #
def halflife(d: pd.Series) -> float:
    """AR(1) half-life of the raw discount, in trading days (NaN if not mean-reverting)."""
    s = pd.Series(d).dropna()
    if len(s) < 100:
        return float("nan")
    y = s.diff().dropna()
    x = s.shift(1).dropna().reindex(y.index)
    xm = x.mean()
    b = float(((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum())
    if b >= 0:
        return float("inf")
    return float(-np.log(2.0) / np.log(1.0 + b))


def mean_reversion_test(panel: dict[str, pd.DataFrame], fwd_days: int = FWD_DAYS) -> dict:
    """Regress the forward change in the discount on today's trailing z, per name and pooled.

    ``d_{t+h} - d_t = a + b * z_t``. Mean reversion means **b < 0**: an unusually wide
    discount narrows over the next ``h`` days. Overlapping windows are handled with a
    Newey-West *t* at ``h`` lags. Pooled stacks every name's observations (so a long name
    weighs more; ``per_name`` shows whether the pooled result is one name's doing).
    """
    per_name = {}
    pairs: list[tuple[pd.Series, pd.Series]] = []
    for name, f in panel.items():
        z = f["z"]
        d = f["d"]
        fwd = d.shift(-fwd_days) - d
        m = z.notna() & fwd.notna()
        if m.sum() < 200:
            per_name[name] = {"n": int(m.sum()), "beta": float("nan"), "t_beta": float("nan"),
                              "halflife_days": halflife(d)}
            continue
        res = hac_ols(fwd[m].to_numpy(), z[m].to_numpy(), lags=fwd_days)
        res["halflife_days"] = halflife(d)
        per_name[name] = res
        pairs.append((fwd[m], z[m]))
    pooled = hac_ols_panel(pairs, lags=int(1.5 * fwd_days)) if pairs else {}
    n_neg = sum(1 for v in per_name.values() if np.isfinite(v.get("beta", np.nan)) and v["beta"] < 0)
    n_fin = sum(1 for v in per_name.values() if np.isfinite(v.get("beta", np.nan)))
    return {"per_name": per_name, "pooled": pooled, "fwd_days": int(fwd_days),
            "n_names_negative_beta": int(n_neg), "n_names": int(n_fin)}


# --------------------------------------------------------------------------- #
# (b) Does buying the wide discount pay?
# --------------------------------------------------------------------------- #
def _positions(panel: dict[str, pd.DataFrame], enter: float, exit_: float,
               always_on: bool) -> pd.DataFrame:
    """{0,1} holding state per name, with hysteresis, then lagged one day.

    ``always_on`` ignores the z entirely and holds every name whenever it is available —
    the structural control. Otherwise a name is entered when ``z >= enter`` and held until
    ``z <= exit_``.
    """
    cols = {}
    for name, f in panel.items():
        z = f["z"]
        if always_on:
            state = pd.Series(np.where(f["d"].notna(), 1.0, np.nan), index=f.index)
        else:
            state = pd.Series(np.nan, index=f.index)
            held = 0.0
            vals = z.to_numpy()
            out = np.full(len(vals), np.nan)
            for i, v in enumerate(vals):
                if not np.isfinite(v):
                    out[i] = np.nan
                    held = 0.0
                    continue
                if held == 0.0 and v >= enter:
                    held = 1.0
                elif held == 1.0 and v <= exit_:
                    held = 0.0
                out[i] = held
            state = pd.Series(out, index=f.index)
        cols[name] = state.shift(1)          # THE one execution lag: signal at t, held over t+1
    return pd.DataFrame(cols)


def run_pairs(panel: dict[str, pd.DataFrame], enter: float = ENTER_Z, exit_: float = EXIT_Z,
              cost_bps: float = COST_BPS, borrow_bps: float = BORROW_BPS,
              always_on: bool = False, hedge: str = "dollar") -> dict:
    """Equal-weighted hedged-pair portfolio; returns are already excess-of-cash.

    Per name and day, the position is long the holdco and short ``h`` of the stake —
    ``h = 1`` for the headline dollar-neutral hedge, ``h = 1/(1-d)`` for the look-through
    variant — **gross-normalised to one unit of notional** and then given an equal share of
    the active names. Costs charge ``cost_bps`` one-way on the change in each leg's
    notional; the short leg accrues ``borrow_bps``/yr on its notional.
    """
    hedge_col = "h" if hedge == "dollar" else "h_lt"
    idx = sorted(set().union(*[f.index for f in panel.values()])) if panel else []
    idx = pd.DatetimeIndex(idx)
    states = _positions(panel, enter, exit_, always_on).reindex(idx)
    n_active = states.fillna(0.0).sum(axis=1)
    weights = states.div(n_active.where(n_active > 0), axis=0).fillna(0.0)

    gross = pd.Series(0.0, index=idx)
    borrow = pd.Series(0.0, index=idx)
    cost = pd.Series(0.0, index=idx)
    short_notional = pd.Series(0.0, index=idx)
    for name, f in panel.items():
        w = weights[name]
        h = f[hedge_col].reindex(idx)
        rh = f["r_holdco"].reindex(idx).fillna(0.0)
        rs = f["r_stake"].reindex(idx).fillna(0.0)
        # h is known at t-1 (it is a price ratio at the close that set the position)
        h_held = h.shift(1).reindex(idx).fillna(0.0)
        # GROSS-NORMALISE. A holdco at a d discount needs 1/(1-d) of stake sold short per
        # unit held, so an unnormalised pair on a 60%-discount name is a 3.5x book. We
        # scale each name's pair to one unit of GROSS notional, split long/short, so a
        # wide discount buys no hidden leverage and the names are comparable.
        gl = 1.0 / (1.0 + h_held.where(h_held > 0))
        long_leg = (w * gl).fillna(0.0)
        short_leg = (w * gl * h_held).fillna(0.0)
        gross = gross + long_leg * rh - short_leg * rs
        short_notional = short_notional + short_leg.abs()
        borrow = borrow + short_leg.abs() * (borrow_bps * 1e-4) / TRADING_DAYS
        long_turn = long_leg.diff().abs().fillna(long_leg.abs())
        short_turn = short_leg.diff().abs().fillna(short_leg.abs())
        cost = cost + (long_turn + short_turn) * (cost_bps * 1e-4)

    net = (gross - borrow - cost).rename("r_net")
    return {
        "gross": gross.rename("r_gross"), "net": net,
        "borrow": borrow, "cost": cost,
        "weights": weights, "n_active": n_active,
        "short_notional": short_notional,
        "summary_gross": summary(gross), "summary_net": summary(net),
        "invested_frac": float((n_active > 0).mean()),
        "avg_names_held": float(n_active[n_active > 0].mean()) if (n_active > 0).any() else 0.0,
        "turnover_ann": float(cost.gt(0).mean() * TRADING_DAYS),
    }


def compare(panel: dict[str, pd.DataFrame], enter: float = ENTER_Z, exit_: float = EXIT_Z,
            cost_bps: float = COST_BPS, borrow_bps: float = BORROW_BPS,
            hedge: str = "dollar") -> dict:
    """Race the timed wide-discount rule against the always-on hedged pair.

    Both arms are hedged long/short pairs, so both are excess-of-cash by construction; the
    difference is *timing on the discount* and nothing else.
    """
    timed = run_pairs(panel, enter, exit_, cost_bps, borrow_bps, always_on=False, hedge=hedge)
    base = run_pairs(panel, enter, exit_, cost_bps, borrow_bps, always_on=True, hedge=hedge)
    diff = (timed["net"] - base["net"]).dropna()
    return {
        "timed": timed, "always_on": base,
        "sharpe_timed": timed["summary_net"]["sharpe"],
        "sharpe_always_on": base["summary_net"]["sharpe"],
        "sharpe_adv": timed["summary_net"]["sharpe"] - base["summary_net"]["sharpe"],
        "t_timed": timed["summary_net"]["tstat"],
        "t_always_on": base["summary_net"]["tstat"],
        "t_diff": newey_west_t(diff.to_numpy()),
        "t_timed_gross": timed["summary_gross"]["tstat"],
    }


# --------------------------------------------------------------------------- #
# The long-only race (explicitly excess-of-cash)
# --------------------------------------------------------------------------- #
def long_only_race(panel: dict[str, pd.DataFrame], cash: pd.Series,
                   enter: float = ENTER_Z, exit_: float = EXIT_Z,
                   cost_bps: float = COST_BPS) -> dict:
    """Unhedged variant: own the holdco when its discount is wide, else cash.

    No short leg, so no borrow — but the position now carries the underlying business's
    direction, which is why the hedged pair is the headline. Raced **excess-of-cash**
    against always owning the panel of holdcos and against always owning the stakes.
    """
    idx = sorted(set().union(*[f.index for f in panel.values()])) if panel else []
    idx = pd.DatetimeIndex(idx)
    r_cash = pd.Series(cash).reindex(idx).ffill().pct_change().fillna(0.0)

    states = _positions(panel, enter, exit_, always_on=False).reindex(idx)
    n_active = states.fillna(0.0).sum(axis=1)
    w = states.div(n_active.where(n_active > 0), axis=0).fillna(0.0)
    avail = _positions(panel, enter, exit_, always_on=True).reindex(idx)
    n_avail = avail.fillna(0.0).sum(axis=1)
    w_all = avail.div(n_avail.where(n_avail > 0), axis=0).fillna(0.0)

    r_timed = pd.Series(0.0, index=idx)
    r_hold = pd.Series(0.0, index=idx)
    r_stake = pd.Series(0.0, index=idx)
    cost = pd.Series(0.0, index=idx)
    for name, f in panel.items():
        rh = f["r_holdco"].reindex(idx).fillna(0.0)
        rs = f["r_stake"].reindex(idx).fillna(0.0)
        r_timed = r_timed + w[name] * rh
        r_hold = r_hold + w_all[name] * rh
        r_stake = r_stake + w_all[name] * rs
        cost = cost + w[name].diff().abs().fillna(w[name].abs()) * (cost_bps * 1e-4)
    # uninvested days sit in cash; excess-of-cash therefore nets the cash leg out
    inv = (n_active > 0).astype(float)
    e_timed = (r_timed + (1.0 - inv) * r_cash - cost - r_cash).rename("e_timed")
    e_hold = (r_hold - r_cash).rename("e_hold")
    e_stake = (r_stake - r_cash).rename("e_stake")
    return {
        "e_timed": e_timed, "e_hold": e_hold, "e_stake": e_stake,
        "timed": summary(e_timed), "holdcos": summary(e_hold), "stakes": summary(e_stake),
        "t_diff_vs_holdcos": newey_west_t((e_timed - e_hold).dropna().to_numpy()),
        "invested_frac": float(inv.mean()),
    }


def cash_cross_check(cash_proxy: pd.Series, bil_total: pd.Series) -> dict:
    """Compare the ^IRX-built cash index with BIL's actual total return over the overlap."""
    idx = cash_proxy.index.intersection(bil_total.index)
    a = cash_proxy.reindex(idx).pct_change().dropna()
    b = bil_total.reindex(idx).pct_change().dropna()
    j = a.index.intersection(b.index)
    a, b = a.loc[j], b.loc[j]
    yrs = len(j) / TRADING_DAYS
    return {"n_days": int(len(j)),
            "irx_cagr": float((1 + a).prod() ** (1 / yrs) - 1) if yrs > 0 else float("nan"),
            "bil_cagr": float((1 + b).prod() ** (1 / yrs) - 1) if yrs > 0 else float("nan"),
            "corr": float(a.corr(b))}


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def cost_sweep(panel, grid=(0.0, 5.0, 10.0, 25.0, 50.0), borrow_bps: float = BORROW_BPS) -> list[dict]:
    rows = []
    for c in grid:
        cmp = compare(panel, cost_bps=c, borrow_bps=borrow_bps)
        rows.append({"cost_bps": c, "sharpe_timed": cmp["sharpe_timed"],
                     "sharpe_always_on": cmp["sharpe_always_on"],
                     "sharpe_adv": cmp["sharpe_adv"], "t_timed": cmp["t_timed"],
                     "t_diff": cmp["t_diff"]})
    return rows


def borrow_sweep(panel, grid=(0.0, 50.0, 100.0, 300.0, 600.0), cost_bps: float = COST_BPS) -> list[dict]:
    rows = []
    for b in grid:
        cmp = compare(panel, cost_bps=cost_bps, borrow_bps=b)
        rows.append({"borrow_bps": b, "sharpe_timed": cmp["sharpe_timed"],
                     "sharpe_always_on": cmp["sharpe_always_on"],
                     "sharpe_adv": cmp["sharpe_adv"], "t_timed": cmp["t_timed"]})
    return rows


def threshold_sweep(panel, enters=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                    exits=(-0.5, 0.0, 0.5), cost_bps: float = COST_BPS,
                    borrow_bps: float = BORROW_BPS) -> list[dict]:
    """Sweep the two thresholds that *define* the rule.

    ``enter`` and ``exit_`` are the study's only genuinely free parameters — costs, borrow,
    ``k`` and the hedge ratio are either priced inputs or derived — so leaving them unswept
    would leave the headline open to the obvious objection that 1.0/0.0 was chosen with
    hindsight. Sweeping them is also the sharpest single robustness statement available: it
    shows the whole grid at once, not just the cell that was reported.
    """
    rows = []
    for e in enters:
        for x in exits:
            if x >= e:
                continue
            cmp = compare(panel, enter=e, exit_=x, cost_bps=cost_bps, borrow_bps=borrow_bps)
            rows.append({"enter": e, "exit": x,
                         "sharpe_timed": cmp["sharpe_timed"], "t_timed": cmp["t_timed"],
                         "sharpe_gross": cmp["timed"]["summary_gross"]["sharpe"],
                         "t_gross": cmp["t_timed_gross"],
                         "invested_frac": cmp["timed"]["invested_frac"]})
    return rows


def zwindow_sweep(px_only, px_total, stakes, windows=(252, 378, 504, 756, 1008),
                  safe=None) -> list[dict]:
    """Sweep the length of the trailing standardisation window (headline: 504 days)."""
    kw = {} if safe is None else {"safe": safe}
    rows = []
    for w in windows:
        panel = build_panel(px_only, px_total, stakes, z_window=w,
                            z_min_periods=min(Z_MIN_PERIODS, w), **kw)
        cmp = compare(panel)
        mr = mean_reversion_test(panel)
        rows.append({"z_window": w, "sharpe_timed": cmp["sharpe_timed"],
                     "t_timed": cmp["t_timed"],
                     "sharpe_gross": cmp["timed"]["summary_gross"]["sharpe"],
                     "t_gross": cmp["t_timed_gross"],
                     "pooled_beta": mr["pooled"].get("beta", float("nan")),
                     "pooled_t": mr["pooled"].get("t_beta", float("nan"))})
    return rows


def calibration_sweep(px_only, px_total, stakes, mults=(0.8, 0.9, 1.0, 1.1, 1.2),
                      other_mults=(0.5, 1.0, 1.5), safe=None) -> list[dict]:
    """Perturb the NAV assumptions and re-run: is the answer an artefact of ``k`` / ``o``?"""
    kw = {} if safe is None else {"safe": safe}
    rows = []
    for km in mults:
        for om in other_mults:
            panel = build_panel(px_only, px_total, stakes, k_mult=km, other_mult=om, **kw)
            cmp = compare(panel)
            mr = mean_reversion_test(panel)
            rows.append({"k_mult": km, "other_mult": om,
                         "sharpe_timed": cmp["sharpe_timed"],
                         "t_timed": cmp["t_timed"],
                         "sharpe_adv": cmp["sharpe_adv"],
                         "pooled_beta": mr["pooled"].get("beta", float("nan")),
                         "pooled_t": mr["pooled"].get("t_beta", float("nan"))})
    return rows


def leave_one_out(panel: dict[str, pd.DataFrame]) -> list[dict]:
    """Drop each name in turn — is the headline one holdco wearing a panel costume?"""
    rows = []
    for drop in list(panel.keys()):
        sub = {k: v for k, v in panel.items() if k != drop}
        cmp = compare(sub)
        mr = mean_reversion_test(sub)
        rows.append({"dropped": drop, "sharpe_timed": cmp["sharpe_timed"],
                     "t_timed": cmp["t_timed"], "sharpe_adv": cmp["sharpe_adv"],
                     "pooled_beta": mr["pooled"].get("beta", float("nan")),
                     "pooled_t": mr["pooled"].get("t_beta", float("nan"))})
    return rows


def era_cut(panel: dict[str, pd.DataFrame], split: str = "2016-01-01") -> dict:
    """Split every name's frame at ``split`` and re-run both tests on each half."""
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = {k: v.loc[sl] for k, v in panel.items() if len(v.loc[sl]) > Z_MIN_PERIODS + 60}
        if not sub:
            out[tag] = None
            continue
        cmp = compare(sub)
        mr = mean_reversion_test(sub)
        out[tag] = {
            "n_names": len(sub),
            "n_days": cmp["timed"]["summary_net"]["n_days"],
            "sharpe_timed": cmp["sharpe_timed"],
            "sharpe_always_on": cmp["sharpe_always_on"],
            "sharpe_adv": cmp["sharpe_adv"],
            "t_timed": cmp["t_timed"],
            "t_diff": cmp["t_diff"],
            "pooled_beta": mr["pooled"].get("beta", float("nan")),
            "pooled_t": mr["pooled"].get("t_beta", float("nan")),
        }
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(frames: dict[str, pd.DataFrame], cost_bps: float = COST_BPS,
                     borrow_bps: float = BORROW_BPS) -> dict:
    """Run both tests on a synthetic panel: planted OU must fire, random walk must not."""
    panel = build_panel_synthetic(frames)
    mr = mean_reversion_test(panel)
    cmp = compare(panel, cost_bps=cost_bps, borrow_bps=borrow_bps)
    return {
        "pooled_beta": mr["pooled"].get("beta", float("nan")),
        "pooled_t": mr["pooled"].get("t_beta", float("nan")),
        "sharpe_timed": cmp["sharpe_timed"],
        "t_timed": cmp["t_timed"],
        "sharpe_timed_gross": cmp["timed"]["summary_gross"]["sharpe"],
        "t_timed_gross": cmp["t_timed_gross"],
        "sharpe_always_on": cmp["sharpe_always_on"],
        "sharpe_adv": cmp["sharpe_adv"],
        "invested_frac": cmp["timed"]["invested_frac"],
        "n_days": cmp["timed"]["summary_net"]["n_days"],
    }
