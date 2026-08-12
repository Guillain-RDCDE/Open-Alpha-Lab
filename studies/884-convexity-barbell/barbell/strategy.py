"""Strategy + inference for Study 884 — Convexity Barbell.

A duration-matched **barbell** (short + long Treasury ends, weighted to the belly's
duration) vs the **bullet** (the belly, IEF). The barbell carries more convexity, so at
equal duration it should out-earn the bullet when yields move a lot — but the market
charges for convexity via a lower yield/carry, and the barbell is exposed to non-parallel
(butterfly) curve moves the bullet is not. This module decides, from the tape, whether
the convexity pickup is a genuine net edge.

Method:

* **Total-return panel.** Per-ETF daily simple returns from adjusted Close (SHY, IEF, TLT,
  BIL).
* **Rates factor & empirical durations.** A common "rates" factor = the equal-weight mean
  of the three bond returns; each bond's **empirical duration** is its trailing-``window``
  beta to that factor (``Cov(r_i,f)/Var(f)``), monotone SHY < IEF < TLT. Value on row ``t``
  uses data through ``t``; the book shifts it one day so a day-``t`` weight uses betas known
  at the close of ``t-1``.
* **The duration match.** Each day, solve the barbell weight ``w`` on SHY (and ``1-w`` on
  TLT) so the barbell's empirical duration equals the bullet's:
  ``w·β_SHY + (1-w)·β_TLT = β_IEF``  ⇒  ``w = (β_TLT − β_IEF)/(β_TLT − β_SHY)``, clamped to
  ``[0,1]``. The barbell and the bullet then share the same first-order rate exposure.
* **The spread.** ``spread = r_barbell − r_bullet`` — the convexity capture net of the
  yield give-up the market charges, at matched duration.
* **Convexity decomposition.** Regress each book's return on ``[1, f, f²]`` to recover its
  duration (``−β₁``) and convexity (``2·β₂``); the barbell's convexity must exceed the
  bullet's. Regress the spread on ``[1, f²]`` to split its mean into a **carry/drift**
  intercept and a **convexity** component (``slope·E[f²]``).
* **Inference.** Newey-West (HAC) *t* on the daily spread; a one-sample *t* and a Welch *t*
  (barbell-excess vs bullet-excess) cross-check; an excess-vs-excess Sharpe race
  (both legs minus BIL); a block-bootstrap Sharpe/mean CI; a two-era robustness cut; a max
  drawdown + calendar-year table; a permutation placebo that breaks the convexity link; a
  costed timer charging turnover on the (rebalanced) barbell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel + rates factor + empirical durations
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple total-return returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def rates_factor(ret: pd.DataFrame, bonds: list[str]) -> pd.Series:
    """Equal-weight mean of the three bond daily returns — the common rates factor."""
    return ret[bonds].mean(axis=1).rename("factor")


def empirical_durations(ret: pd.DataFrame, bonds: list[str],
                        window: int = 252) -> pd.DataFrame:
    """Trailing ``window``-day beta of each bond to the rates factor (empirical duration).

    ``β_i = Cov(r_i, f)/Var(f)`` over the trailing window (value on row ``t`` uses returns
    through ``t`` inclusive). Vectorised across dates; the loop is over the three bond
    columns only."""
    f = rates_factor(ret, bonds)
    var_f = f.rolling(window, min_periods=window).var()
    out = {}
    for c in bonds:
        cov = ret[c].rolling(window, min_periods=window).cov(f)
        out[c] = cov / var_f
    betas = pd.DataFrame(out, index=ret.index)
    bad = np.broadcast_to((var_f <= 0).to_numpy()[:, None], betas.shape)
    return betas.mask(pd.DataFrame(bad, index=betas.index, columns=betas.columns))


# --------------------------------------------------------------------------- #
# The duration-matched barbell vs the bullet
# --------------------------------------------------------------------------- #
def match_weight(beta_short: np.ndarray, beta_belly: np.ndarray,
                 beta_long: np.ndarray) -> np.ndarray:
    """Barbell weight on the short leg so the barbell duration == belly duration.

    ``w·β_short + (1-w)·β_long = β_belly`` ⇒ ``w = (β_long−β_belly)/(β_long−β_short)``,
    clamped to ``[0,1]``. Vectorised over the date axis."""
    denom = beta_long - beta_short
    w = np.where(np.abs(denom) > 1e-9, (beta_long - beta_belly) / denom, np.nan)
    return np.clip(w, 0.0, 1.0)


def barbell_book(ret: pd.DataFrame, bonds: list[str], cash: str,
                 window: int = 252) -> pd.DataFrame:
    """Daily duration-matched barbell vs bullet.

    ``bonds`` are ordered [short, belly, long] (SHY, IEF, TLT). Each day ``t`` the barbell
    weight is built from empirical durations known at the close of ``t-1`` (one ``shift``);
    the barbell return is ``w·r_short + (1-w)·r_long``, the bullet return is ``r_belly``,
    and the spread is their difference. Also returns the cash leg (for excess races) and
    the per-day target weights (for the costed timer). Fully vectorised across dates.
    """
    short, belly, long = bonds
    B = empirical_durations(ret, bonds, window).shift(1)     # known at t-1
    w = match_weight(B[short].to_numpy(), B[belly].to_numpy(), B[long].to_numpy())

    R = ret
    r_short, r_belly, r_long = R[short].to_numpy(), R[belly].to_numpy(), R[long].to_numpy()
    r_cash = R[cash].to_numpy() if cash in R.columns else np.zeros(len(R))

    r_barbell = w * r_short + (1.0 - w) * r_long
    r_bullet = r_belly
    spread = r_barbell - r_bullet

    out = pd.DataFrame(
        {
            "r_barbell": r_barbell, "r_bullet": r_bullet, "spread": spread,
            "r_cash": r_cash, "w_short": w, "w_long": 1.0 - w,
            "r_short": r_short, "r_long": r_long,
        },
        index=ret.index,
    )
    valid = np.isfinite(w) & np.isfinite(r_barbell) & np.isfinite(r_bullet) & np.isfinite(r_cash)
    out = out[valid]
    out.attrs["factor"] = rates_factor(ret, bonds).reindex(out.index)
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (shared house kit — mirror of study 803 / 826)
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


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def annualized_sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, block: int = 20,
                      seed: int = 884, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the mean (bps/day) of a daily series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < block + 1:
        return {"lo_bps": float("nan"), "hi_bps": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    xx = np.concatenate([x, x[:block]])          # wrap for circular blocks
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = xx[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"lo_bps": float(lo * 1e4), "hi_bps": float(hi * 1e4), "n": n}


# --------------------------------------------------------------------------- #
# Convexity decomposition
# --------------------------------------------------------------------------- #
def convexity_capture(book: pd.DataFrame) -> dict:
    """Duration-match check + convexity slope + the spread's carry/convexity split.

    Regress the daily spread on ``[1, f, f²]`` (``f`` = rates factor). Because the two
    books are duration matched, the **linear** coefficient (β₁) should be ≈ 0 (a residual
    duration check). The **quadratic** coefficient (β₂ > 0) is the convexity signature:
    the barbell out-earns the bullet more the bigger the squared rate move. The mean spread
    then splits into a **carry/drift** part (the part not explained by the squared move,
    ``mean − β₂·E[f²]``) and a **convexity** part (``β₂·E[f²]``).

    Note: the return-based factor is a near-linear proxy for the true yield change
    (``corr ≈ −0.99``), so β₂ recovers the *sign and relative size* of the convexity
    pickup, not an absolute analytical convexity — the honest thing a returns-only tape
    can measure."""
    f = book.attrs["factor"].to_numpy(dtype=float)
    sp = book["spread"].to_numpy(dtype=float)
    m = np.isfinite(sp) & np.isfinite(f)
    sp, f = sp[m], f[m]
    if len(sp) < 5:
        return {"resid_dur_slope": float("nan"), "conv_slope": float("nan"),
                "spread_carry_bps": float("nan"), "spread_conv_bps": float("nan")}
    f2 = f ** 2
    X = np.column_stack([np.ones(len(f)), f, f2])
    coef, *_ = np.linalg.lstsq(X, sp, rcond=None)
    ef2 = float(f2.mean())
    conv_slope = float(coef[2])
    conv_bps = float(conv_slope * ef2 * 1e4)          # convexity component of the mean
    mean_bps = float(sp.mean() * 1e4)
    return {
        "resid_dur_slope": float(coef[1]),            # ~0 if duration matched
        "conv_slope": conv_slope,                     # >0 = barbell more convex
        "spread_conv_bps": conv_bps,                  # convexity part of the mean spread
        "spread_carry_bps": mean_bps - conv_bps,      # carry/drift part of the mean spread
    }


def convexity_smile(book: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Mean spread (bps/day) bucketed by the size of the rate move.

    Days are grouped by absolute rates-factor move ``|f|`` into ``n_bins`` quantile
    buckets; a convex barbell out-earns the bullet more in the big-move buckets — the
    classic convexity smile (here a monotone rise in ``|move|``)."""
    f = book.attrs["factor"].reindex(book.index).to_numpy(dtype=float)
    sp = book["spread"].to_numpy(dtype=float)
    m = np.isfinite(sp) & np.isfinite(f)
    af, sp = np.abs(f[m]), sp[m]
    q = np.quantile(af, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    b = np.digitize(af, q[1:-1])
    rows = []
    for i in range(n_bins):
        sel = sp[b == i]
        rows.append({"bucket": i, "n": int(sel.size),
                     "mean_spread_bps": float(sel.mean() * 1e4) if sel.size else float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Drawdown + calendar-year table
# --------------------------------------------------------------------------- #
def max_drawdown(r: np.ndarray) -> float:
    """Max drawdown (fraction, negative) of the compounded series of daily returns."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def calendar_year_table(book: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-year total return (%) for barbell, bullet, and their spread."""
    g = book[["r_barbell", "r_bullet", "spread"]].copy()
    g["year"] = g.index.year

    def _comp(col):
        return g.groupby("year")[col].apply(lambda s: (np.prod(1.0 + s.to_numpy()) - 1.0) * 100)

    out = pd.DataFrame({
        "barbell_%": _comp("r_barbell"),
        "bullet_%": _comp("r_bullet"),
        "spread_%": g.groupby("year")["spread"].apply(lambda s: s.sum() * 100),
    })
    return out


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def barbell_stats(book: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = book["spread"].to_numpy(dtype=float)
    barx = book["r_barbell"].to_numpy() - book["r_cash"].to_numpy()   # excess of cash
    bulx = book["r_bullet"].to_numpy() - book["r_cash"].to_numpy()
    cc = convexity_capture(book)
    out = {
        "n_days": int(len(book)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "spread_sharpe": annualized_sharpe(sp),
        "welch_t": welch_t(barx, bulx),
        # excess-vs-excess Sharpe race (both minus BIL)
        "sharpe_barbell_x": annualized_sharpe(barx),
        "sharpe_bullet_x": annualized_sharpe(bulx),
        "sharpe_adv": annualized_sharpe(barx) - annualized_sharpe(bulx),
        # drawdowns
        "mdd_barbell": max_drawdown(book["r_barbell"].to_numpy()),
        "mdd_bullet": max_drawdown(book["r_bullet"].to_numpy()),
        # mean weights
        "w_short": float(np.nanmean(book["w_short"].to_numpy())),
        "w_long": float(np.nanmean(book["w_long"].to_numpy())),
    }
    out.update(cc)
    return out


# --------------------------------------------------------------------------- #
# Placebo — is the convexity spread real, or a lucky leg alignment?
# --------------------------------------------------------------------------- #
def placebo_pvalue(ret: pd.DataFrame, bonds: list[str], cash: str,
                   window: int = 252, n_seeds: int = 20, n_draws_per_seed: int = 50,
                   base_seed: int = 884) -> dict:
    """Keep the duration-matched barbell weights, but read the two barbell legs' returns
    from a **time-permuted** copy (each leg's own return block-shuffled the same way),
    breaking the day-by-day alignment of the big-move convexity term while preserving each
    leg's marginal distribution. p = share of permuted worlds whose spread mean >= observed
    (right-tail; the claim predicts a positive convexity spread)."""
    book = barbell_book(ret, bonds, cash, window)
    obs = float(book["spread"].mean())
    w = book["w_short"].to_numpy()
    r_short = book["r_short"].to_numpy()
    r_long = book["r_long"].to_numpy()
    r_bullet = book["r_bullet"].to_numpy()
    n = len(book)

    means = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed)
        for _ in range(n_draws_per_seed):
            p1 = rng.permutation(n)
            p2 = rng.permutation(n)
            r_bar = w * r_short[p1] + (1.0 - w) * r_long[p2]
            means.append(np.nanmean(r_bar - r_bullet))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4),
        "p_value": float((means >= obs).mean()),
        "n_draws": len(means),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer — turnover on the rebalanced barbell
# --------------------------------------------------------------------------- #
def timer_stats(book: pd.DataFrame, cost_bps: float = 1.0) -> dict:
    """Cost the barbell relative to the buy-and-hold bullet.

    The bullet is 100% IEF (buy-and-hold, ~zero turnover). The barbell holds two legs and
    must be rebalanced back to the day's duration-match target — turnover comes from both
    the re-estimated target drifting and the two legs drifting apart intraday. We charge a
    one-way cost on that daily L1 turnover. (The bullet pays no rebalancing cost.)"""
    w = book["w_short"].to_numpy(dtype=float)
    r_short = book["r_short"].to_numpy(dtype=float)
    r_long = book["r_long"].to_numpy(dtype=float)
    n = len(w)
    # end-of-day drifted short weight given start weight w[t] and the two leg returns
    num = w * (1.0 + r_short)
    den = num + (1.0 - w) * (1.0 + r_long)
    drift_short = np.where(den > 0, num / den, w)
    # turnover at t = |w[t] - drift_short[t-1]| on the short leg, doubled for both legs
    turn = np.empty(n)
    turn[0] = abs(w[0]) + abs(1.0 - w[0])           # build the book on day 1
    turn[1:] = 2.0 * np.abs(w[1:] - drift_short[:-1])
    cost = (cost_bps / 1e4) * turn
    gross = book["spread"].to_numpy(dtype=float)
    net = gross - cost
    sd = np.nanstd(net, ddof=1) if n > 1 else float("nan")
    sharpe = np.nanmean(net) / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": float(np.nanmean(gross) * 1e4),
        "net_bps": float(np.nanmean(net) * 1e4),
        "cost_bps_per_day": float(np.nanmean(cost) * 1e4),
        "avg_turnover": float(np.nanmean(turn)),
        "ann_net_pct": float(np.nanmean(net) * TRADING_DAYS * 100),
        "sharpe_net": float(sharpe),
        "t_net": one_sample_t(net),
        "t_nw_net": newey_west_t(net, 10),
    }


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_table(book: pd.DataFrame, split: str = "2018-01-01") -> list[dict]:
    """Two-era robustness cut on the spread (mean bps + NW t)."""
    rows = []
    for lo, hi, lbl in [("2000-01-01", split, f"< {split}"),
                        (split, "2100-01-01", f">= {split}")]:
        sub = book[(book.index >= lo) & (book.index < hi)]
        sp = sub["spread"].to_numpy(dtype=float)
        rows.append({
            "era": lbl, "n": len(sub),
            "spread_bps": float(np.nanmean(sp) * 1e4) if len(sp) else float("nan"),
            "t_nw": newey_west_t(sp, 10) if len(sp) else float("nan"),
            "sharpe": annualized_sharpe(sp) if len(sp) else float("nan"),
        })
    return rows


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 252) -> dict:
    """Run the headline barbell stats on a synthetic panel."""
    from . import data as _d
    ret = close_returns(panel)
    book = barbell_book(ret, _d.BOND_TICKERS, _d.CASH_TICKER, window)
    ts = barbell_stats(book)
    return {
        "spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
        "spread_sharpe": ts["spread_sharpe"],
        "conv_slope": ts["conv_slope"], "resid_dur_slope": ts["resid_dur_slope"],
        "spread_conv_bps": ts["spread_conv_bps"], "spread_carry_bps": ts["spread_carry_bps"],
        "sharpe_adv": ts["sharpe_adv"], "n_days": ts["n_days"],
    }
