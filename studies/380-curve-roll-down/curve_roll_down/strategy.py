"""Strategy + inference for Study 380 — Curve-Roll-Down (riding the yield curve).

The believers' rule:

    Buy a bond at the sleeve maturity (here 5y) on a **steep** curve. Hold a year.
    If the curve is unchanged the bond *ages* down the curve to a lower-yield point;
    because price moves inversely to yield, you collect the **carry** (its yield) PLUS
    a **roll-down** price gain. The pitch: this beats holding cash, reliably.

We turn this into a measurable, no-look-ahead return and test it honestly:

  * **Expected (textbook) roll+carry.** From today's curve alone:
    ``carry = sleeve_yield`` and ``roll = duration * (sleeve_yield - rolled_yield)``.
    This is what the brochure promises — computed from the curve, before any rate move.

  * **Realized total return of the duration sleeve.** A 1-year duration approximation:
    ``ret ≈ y0  -  D * Δy`` where ``y0`` is the entry sleeve yield (carry) and ``Δy`` is
    the *actual* change in the sleeve-maturity yield over the year (the curve moving).
    The realized **excess over cash** subtracts the average short rate held.

  * **Inference.** Newey-West / HAC ``t`` of the realized excess return (overlapping
    annual windows ⇒ serial correlation, so a plain t is wrong); a **placebo** null on
    the slope-timing overlay; a **win-rate** vs the cash base rate (50% by construction
    of "excess > 0"); a **1-day execution lag**; and **one-way costs** on the timing
    overlay's turnover.

The decisive question is not "is roll+carry positive on a steep curve" — mechanically it
is. It is whether the *realized* excess survives the curve **moving against you** (a
rising-rate year erases the roll-down and then some), and whether the slope adds anything
beyond simply **being long duration** (the term premium). A static-curve free lunch that a
single hiking cycle wipes out is carry dressed as alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_YEAR = 252


# --------------------------------------------------------------------------- #
# Curve mechanics
# --------------------------------------------------------------------------- #
def sleeve_duration(sleeve_mat: float = 5.0) -> float:
    """A crude modified duration for a par sleeve of maturity ``sleeve_mat`` (years)."""
    return max(sleeve_mat - 0.5, 0.5)


def expected_roll_carry(frame: pd.DataFrame, sleeve_mat: float = 5.0) -> pd.Series:
    """The textbook *expected* 1-year roll+carry from today's curve (in return units).

    ``carry`` = the sleeve yield you earn; ``roll`` = duration * (sleeve - rolled) yield
    pickup as the bond ages one year down a *static* curve. Returns a fraction (e.g. 0.05).
    """
    D = sleeve_duration(sleeve_mat)
    carry = frame["sleeve"] / 100.0
    roll = D * (frame["sleeve"] - frame["rolled"]) / 100.0
    return carry + roll


def realized_excess(frame: pd.DataFrame, sleeve_mat: float = 5.0,
                    horizon: int = TRADING_YEAR, lag: int = 1) -> pd.DataFrame:
    """Realized 1-year **excess-over-cash** return of the duration sleeve, no look-ahead.

    Entry is the close ``lag`` days after the curve is observed. The 1-year sleeve total
    return is approximated as ``y0 - D * Δy`` where ``y0`` is the entry sleeve yield
    (carry) and ``Δy`` is the realized change in the sleeve-maturity yield over the
    horizon (the curve actually moving). Excess subtracts the entry short rate (cash).

    Returns a frame (indexed by entry date) with columns:
      ``entry_yield, exit_yield, dy, carry, price, total, cash, excess, slope, exp_rc``.
    Rows whose horizon overruns the tape are dropped.
    """
    D = sleeve_duration(sleeve_mat)
    sleeve = frame["sleeve"].to_numpy()
    short = frame["short"].to_numpy()
    slope = frame["slope"].to_numpy()
    exp_rc = expected_roll_carry(frame, sleeve_mat).to_numpy()
    idx = frame.index
    n = len(frame)

    rows = []
    for i in range(n):
        entry = i + lag
        exit_ = entry + horizon
        if exit_ >= n:
            break
        y0 = sleeve[entry] / 100.0
        y1 = sleeve[exit_] / 100.0
        dy = y1 - y0
        carry = y0
        price = -D * dy                         # duration P&L from the yield move
        total = carry + price
        cash = short[entry] / 100.0
        rows.append((idx[entry], sleeve[entry], sleeve[exit_], dy * 100.0,
                     carry, price, total, cash, total - cash, slope[entry],
                     exp_rc[entry]))
    out = pd.DataFrame(
        rows,
        columns=["entry_date", "entry_yield", "exit_yield", "dy", "carry",
                 "price", "total", "cash", "excess", "slope", "exp_rc"],
    ).set_index("entry_date")
    return out


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int = TRADING_YEAR) -> float:
    """HAC (Newey-West) t-stat of ``mean(x) = 0`` for an overlapping series.

    Overlapping annual windows induce strong autocorrelation; a naive t overstates
    significance by ~sqrt(overlap). We use a Bartlett-kernel long-run variance with
    ``lags`` lags (default one year) — the honest standard error for this object.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    s = gamma0
    L = min(lags, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        cov = (e[k:] @ e[:-k]) / n
        s += 2.0 * w * cov
    if s <= 0:
        return float("nan")
    se = np.sqrt(s / n)
    if se == 0:
        return float("nan")
    return float(mu / se)


def naive_t(x: np.ndarray) -> float:
    """A plain (i.i.d.-assuming) t of ``mean(x)=0`` — shown only to expose HAC shrinkage."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se else float("nan")


def summarize_excess(res: pd.DataFrame) -> dict:
    """Headline stats on the realized excess-over-cash series."""
    xs = res["excess"].to_numpy()
    return {
        "n": int(len(xs)),
        "mean_excess": float(np.nanmean(xs)),
        "win": float(np.nanmean(xs > 0)),
        "vol": float(np.nanstd(xs, ddof=1)),
        "sharpe": float(np.nanmean(xs) / np.nanstd(xs, ddof=1)) if np.nanstd(xs, ddof=1) else float("nan"),
        "t_hac": newey_west_t(xs),
        "t_naive": naive_t(xs),
        "mean_exp_rc": float(np.nanmean(res["exp_rc"].to_numpy())),
        "mean_dy": float(np.nanmean(res["dy"].to_numpy())),
    }


def promise_vs_reality(res: pd.DataFrame) -> dict:
    """Compare the brochure (expected roll+carry) to the realized excess.

    The gap is the curve *moving* — roll-down assumes a static curve; reality hikes."""
    exp_rc = res["exp_rc"].to_numpy()
    realized_total = res["total"].to_numpy()
    return {
        "mean_promised": float(np.nanmean(exp_rc)),
        "mean_realized_total": float(np.nanmean(realized_total)),
        "mean_gap": float(np.nanmean(exp_rc - realized_total)),
        "frac_realized_below_promise": float(np.nanmean(realized_total < exp_rc)),
    }


def slope_timing_placebo(res: pd.DataFrame, n_draws: int = 20_000,
                         seed: int = 380) -> dict:
    """Does conditioning on a STEEP curve beat random timing?

    The slope-timing overlay holds the sleeve only when the entry slope is in its top
    tercile (its steepest third); the placebo draws the same number of random entry dates
    and asks how often random timing's mean excess matches/beats the steep-curve set.
    """
    xs = res["excess"].to_numpy()
    slope = res["slope"].to_numpy()
    n = len(xs)
    if n < 10:
        return {"k": 0, "steep_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    thr = np.nanquantile(slope, 2.0 / 3.0)
    mask = slope >= thr
    obs = xs[mask]
    k = int(mask.sum())
    obs_mean = float(np.nanmean(obs))
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.integers(0, n, size=k)
        means[i] = np.nanmean(xs[pick])
    p = float((means >= obs_mean).mean())
    return {"k": k, "steep_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def net_of_costs(res: pd.DataFrame, cost_bps: float = 2.0,
                 turnover_per_year: float = 1.0) -> dict:
    """Apply one-way costs to the always-on duration sleeve (annual roll = 1 round-trip).

    The sleeve is rolled once a year (it ages out of its maturity bucket), so turnover is
    ~1x/yr; a one-way cost of ``cost_bps`` is charged on that turnover. (Costs are tiny
    relative to the curve-move risk — the binding constraint is the rate path, not the bid.)
    """
    xs = res["excess"].to_numpy()
    gross = float(np.nanmean(xs))
    c = (cost_bps / 1e4) * turnover_per_year
    return {"n": int(len(xs)), "gross_mean": gross, "net_mean": gross - c,
            "cost_bps": cost_bps}


def synthetic_edge_test(syn: pd.DataFrame, sleeve_mat: float = 5.0,
                        warmup: int = 252) -> dict:
    """Positive control: does the slope predict excess return BEYOND the carry baseline?

    The synthetic excess is ``baseline_carry + edge*(slope_rank-0.5) + noise``. A steep
    curve *mechanically* carries more, so the right null test removes the carry baseline
    first and asks whether the **residual** still rises with the slope. We regress the
    carry-stripped excess on the (lagged, centred) slope rank and report the HAC t of the
    slope coefficient.

    With ``edge = 0`` the residual carries no slope signal ⇒ t must NOT clear 2 (a bare
    carry baseline is not an edge). With a large planted ``edge`` ⇒ t must light up. This
    is the faithful-engine + power control, on data where we know the truth.
    """
    syn = syn.iloc[warmup:].copy()
    slope = syn["slope"].to_numpy()
    D = sleeve_duration(sleeve_mat)
    carry = slope / 100.0
    rolldown = (slope / sleeve_mat) * D / 100.0
    baseline = carry + rolldown                     # the mechanical carry baseline
    resid = syn["xs_fwd1y"].to_numpy() - baseline   # excess stripped of carry
    # lagged, centred slope rank as the regressor
    rank = pd.Series(slope, index=syn.index).rolling(252, min_periods=63).rank(pct=True)
    rank = rank.shift(1).to_numpy() - 0.5
    m = ~np.isnan(rank) & ~np.isnan(resid)
    x, y = rank[m], resid[m]
    x = x - x.mean()
    beta = float((x @ y) / (x @ x)) if (x @ x) else float("nan")
    # HAC t of the slope coefficient via the regression-residual score
    e = y - beta * x
    score = x * e
    t_naive_den = np.sqrt(((e ** 2).sum() / (x @ x)))
    # Newey-West long-run variance of the score for the HAC SE of beta
    n = len(x)
    s2 = (score @ score) / n
    L = min(TRADING_YEAR, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        s2 += 2.0 * w * (score[k:] @ score[:-k]) / n
    se_hac = np.sqrt(n * s2) / (x @ x)
    t_hac = float(beta / se_hac) if se_hac else float("nan")
    return {
        "n": int(n), "beta": beta, "t_hac": t_hac,
        "steep_excess_resid": float(y[(x > x.std())].mean()) if (x > x.std()).any() else float("nan"),
    }


def regime_split(res: pd.DataFrame) -> dict:
    """Realized excess in falling-rate vs rising-rate years (the curve moving for/against).

    Roll-down is a static-curve concept; this splits realized excess by whether the
    sleeve yield FELL (tailwind, roll-down realized) or ROSE (headwind, roll-down erased)."""
    dy = res["dy"].to_numpy()
    xs = res["excess"].to_numpy()
    down = xs[dy < 0]
    up = xs[dy >= 0]
    return {
        "n_down": int(len(down)), "mean_down": float(np.nanmean(down)) if len(down) else float("nan"),
        "n_up": int(len(up)), "mean_up": float(np.nanmean(up)) if len(up) else float("nan"),
    }
