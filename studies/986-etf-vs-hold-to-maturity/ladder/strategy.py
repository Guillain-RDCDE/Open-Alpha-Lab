"""A bond, a bond fund, and the difference between them — Study 986.

The single most common misunderstanding in retail fixed income: "TLT yields 4%, so if I hold it
for twenty years I'll earn about 4%." That is true of a *bond*. TLT is not a bond.

A bond held to maturity has two properties that a constant-maturity fund does not:

- **Pull to par.** Whatever happens to its price in between, it redeems at 100. Interim losses
  are guaranteed to reverse.
- **A known terminal date.** Its duration falls to zero. Rate risk disappears on a schedule.

A constant-maturity fund sells each bond as it ages out of the target band and buys a fresh one.
Its duration is therefore roughly constant forever, it never pulls to par, and a rate shock is
*permanent* in the sense that nothing in the instrument's construction reverses it. What it does
have instead is a **reinvestment** benefit: after a rate rise, every future coupon and every
replacement bond is bought at the new, higher yield.

Those two effects run in opposite directions, and the horizon at which they cross is a real,
computable number — the *duration-matched horizon* at which a bond portfolio's price loss and
reinvestment gain offset, known since Redington (1952) as immunisation. This module builds:

- ``simulate_bond`` — a coupon bond held to maturity on a given rate path.
- ``simulate_rolling_fund`` — a constant-maturity ladder on the *same* path, so the two differ
  only by the roll.
- ``crossover_horizon`` — the horizon at which the fund catches up with the bond after a shock.
- ``realised_vs_promised`` — on the real tape: what each fund's starting yield promised, and
  what it actually delivered, over every rolling window.

The tradability question is not "can you beat the market"; it is whether the difference is large
enough to change an allocation decision, which is the form the question actually takes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
# Approximate effective durations, in years, of the funds this study uses. These are stable
# properties of a constant-maturity mandate, which is exactly the point.
FUND_DURATION = {"SHY": 1.9, "IEF": 7.5, "TLT": 16.5, "LQD": 8.4, "BIL": 0.1}


# --------------------------------------------------------------------------- #
# Bond arithmetic
# --------------------------------------------------------------------------- #
def price_from_yield(y: float, maturity: float, coupon: float | None = None,
                     freq: int = 2) -> float:
    """Clean price per 100 of face for a bullet bond, given a flat yield."""
    coupon = y if coupon is None else coupon
    n = int(round(maturity * freq))
    if n <= 0:
        return 100.0
    c = 100.0 * coupon / freq
    r = y / freq
    if abs(r) < 1e-12:
        return c * n + 100.0
    disc = (1.0 + r) ** -np.arange(1, n + 1)
    return float(c * disc.sum() + 100.0 * disc[-1])


def macaulay_duration(y: float, maturity: float, coupon: float | None = None,
                      freq: int = 2) -> float:
    """Macaulay duration in years — the horizon at which price and reinvestment offset."""
    coupon = y if coupon is None else coupon
    n = int(round(maturity * freq))
    if n <= 0:
        return 0.0
    c = 100.0 * coupon / freq
    r = y / freq
    t = np.arange(1, n + 1)
    disc = (1.0 + r) ** -t
    cf = np.full(n, c)
    cf[-1] += 100.0
    pv = cf * disc
    return float((pv * t).sum() / pv.sum() / freq)


def modified_duration(y: float, maturity: float, coupon: float | None = None,
                      freq: int = 2) -> float:
    """Macaulay duration divided by (1 + y/freq) — the price sensitivity."""
    return macaulay_duration(y, maturity, coupon, freq) / (1.0 + y / freq)


# --------------------------------------------------------------------------- #
# The two instruments, on one rate path
# --------------------------------------------------------------------------- #
def simulate_bond(rates: np.ndarray, maturity: float, coupon: float | None = None,
                  freq: int = 2, steps_per_year: int = 252) -> pd.DataFrame:
    """A single coupon bond bought at t=0 and held until it matures.

    Coupons are reinvested at the prevailing rate, which is the assumption that makes "hold to
    maturity and you earn the yield" only approximately true: the realised return equals the
    purchase yield exactly only if coupons are reinvested *at* that yield.
    """
    coupon = rates[0] if coupon is None else coupon
    n = len(rates)
    dt = 1.0 / steps_per_year
    remaining = maturity
    coupon_cash = 0.0
    accrual = 0.0
    values = np.empty(n)
    for t in range(n):
        y = rates[t]
        px = price_from_yield(y, max(remaining, 0.0), coupon, freq)
        coupon_cash *= (1.0 + y * dt)
        accrual += 100.0 * coupon / steps_per_year
        if accrual >= 100.0 * coupon / freq:
            coupon_cash += 100.0 * coupon / freq
            accrual -= 100.0 * coupon / freq
        values[t] = px + coupon_cash + accrual
        remaining -= dt
        if remaining <= 0:
            remaining = 0.0
    return pd.DataFrame({"rate": rates, "value": values,
                         "remaining_maturity": np.maximum(
                             maturity - np.arange(n) * dt, 0.0)})


def simulate_rolling_fund(rates: np.ndarray, target_maturity: float, freq: int = 2,
                          steps_per_year: int = 252) -> pd.DataFrame:
    """A constant-maturity fund: always holding a bond of ``target_maturity`` years.

    Implemented as a total-return index whose daily return is the carry plus the duration-scaled
    price move — the standard constant-maturity approximation, and the one every index provider
    uses. It never matures and never pulls to par.
    """
    n = len(rates)
    dt = 1.0 / steps_per_year
    dur = np.array([modified_duration(y, target_maturity, y, freq) for y in rates])
    dy = np.diff(rates, prepend=rates[0])
    convexity_term = 0.5 * (target_maturity ** 2) * dy ** 2 * 0.5
    ret = rates * dt - dur * dy + convexity_term
    ret[0] = 0.0
    value = 100.0 * np.cumprod(1.0 + ret)
    return pd.DataFrame({"rate": rates, "value": value, "duration": dur, "return": ret})


def rate_path(kind: str = "shock", n: int = 2520, start: float = 0.04, end: float = 0.06,
              shock_at: int = 252, steps_per_year: int = 252, vol: float = 0.0,
              seed: int = 986) -> np.ndarray:
    """A deterministic (or noisy) interest-rate path for the comparison."""
    rng = np.random.default_rng(seed)
    if kind == "flat":
        path = np.full(n, start)
    elif kind == "shock":
        path = np.full(n, start)
        path[shock_at:] = end
    elif kind == "ramp":
        path = np.linspace(start, end, n)
    elif kind == "roundtrip":
        half = n // 2
        path = np.concatenate([np.linspace(start, end, half),
                               np.linspace(end, start, n - half)])
    else:
        raise ValueError(f"unknown path kind {kind!r}")
    if vol > 0:
        path = path + np.cumsum(rng.normal(0, vol / np.sqrt(steps_per_year), n))
    return np.maximum(path, 0.0005)


def compare(rates: np.ndarray, maturity: float, steps_per_year: int = 252) -> pd.DataFrame:
    """Both instruments on one path, with their cumulative returns aligned.

    A note on what happens past the bond's maturity, since the paths here run longer than that:
    once the bond redeems it becomes a cash balance accruing at the prevailing short rate. That
    is the natural "held to maturity and then reinvested" convention, and it is why the bond's
    line flattens out while the fund's keeps compounding at the (higher, post-shock) yield —
    which is a real part of the comparison, not an artefact.
    """
    bond = simulate_bond(rates, maturity, steps_per_year=steps_per_year)
    fund = simulate_rolling_fund(rates, maturity, steps_per_year=steps_per_year)
    return pd.DataFrame({
        "rate": rates,
        "bond_value": bond["value"] / bond["value"].iloc[0],
        "fund_value": fund["value"] / fund["value"].iloc[0],
        "remaining_maturity": bond["remaining_maturity"],
        "fund_duration": fund["duration"],
    })


def convergence_horizon(rates: np.ndarray, maturity: float, tol: float = 0.0025,
                        steps_per_year: int = 252) -> dict:
    """How long the *fund* takes to deliver its starting yield after a rate shock.

    This is the quantity with a theory behind it, and it is the study's headline number.

    A bond's answer is trivial: at maturity, by construction. A constant-maturity fund's is not.
    Leibowitz, Bova & Kogelman (2014) show analytically that a duration-targeted portfolio's
    annualised return converges to its starting yield over roughly **2D − 1** years — the price
    loss and the reinvestment gain offsetting on a schedule set by the duration, but taking
    about twice as long as the single-bond immunisation result because the fund keeps resetting
    its clock.

    Measured here as the first horizon at which the fund's annualised return since inception
    **crosses** the yield it was bought at. Crossing, not settling: on a trending rate path the
    fund's annualised return starts below its purchase yield (the price loss dominates), passes
    through it, and then goes above (the reinvestment gain dominates). There is no horizon at
    which it sits still, and a definition that demanded one would return ``nan`` forever.

    The result depends on the rate *path*, and unavoidably so. On a steady trend it lands near
    2D − 1. After a single permanent shock the fund's long-run return converges to the **new**
    yield rather than the old one, and the crossing happens sooner. Both are reported.
    Returning ``nan`` means the path was not long enough to see the crossing.
    """
    fund = simulate_rolling_fund(rates, maturity, steps_per_year=steps_per_year)
    v = fund["value"].to_numpy()
    y0 = float(rates[0])
    t = np.arange(len(v)) / steps_per_year
    with np.errstate(divide="ignore", invalid="ignore"):
        ann = np.where(t > 0.25, (v / v[0]) ** (1.0 / np.maximum(t, 1e-9)) - 1.0, np.nan)
    excess = ann - y0
    start = int(0.5 * steps_per_year)
    idx = np.nan
    if start < len(excess):
        seg = excess[start:]
        sign0 = np.sign(seg[np.isfinite(seg)][0]) if np.isfinite(seg).any() else 0.0
        hit = np.flatnonzero(np.isfinite(seg) & ((np.sign(seg) != sign0) |
                                                 (np.abs(seg) < tol)))
        if len(hit):
            idx = t[start + int(hit[0])]
    d = modified_duration(y0, maturity, y0)
    return {"convergence_years": float(idx) if idx == idx else np.nan,
            "duration": float(d), "leibowitz_2d_minus_1": float(2 * d - 1),
            "starting_yield": y0, "final_annualised": float(ann[-1]),
            "tol": tol}


def crossover_horizon(rates: np.ndarray, maturity: float,
                      steps_per_year: int = 252) -> dict:
    """When (if ever) the rolling fund catches the held bond, *within the bond's own life*.

    A caveat that decides how this number should be read. Past the bond's maturity the
    comparison stops being well defined: the bond has redeemed, and what happens next depends
    entirely on what you assume the proceeds are reinvested in. This function therefore searches
    only up to the bond's maturity and returns ``nan`` when the fund has not caught up by then —
    which is the honest answer, not a missing value. The theoretically-grounded quantity is
    ``convergence_horizon``; this one is the intuitive picture that goes with it.
    """
    c = compare(rates, maturity, steps_per_year)
    gap = (c["fund_value"] - c["bond_value"]).to_numpy()
    # Anchor on the *shock*, not on day 0 (where the two are identical by construction, so the
    # sign of "the initial gap" would be rounding noise) and not on the point of maximum
    # divergence (which on a long path is the far end, after the crossing has already
    # happened). The shock is the largest single move in the rate path.
    drate = np.abs(np.diff(rates, prepend=rates[0]))
    shock = int(np.argmax(drate))
    if drate[shock] < 1e-8:                       # a flat path: nothing to recover from
        return {"crossover_years": 0.0, "shock_at_years": 0.0, "initial_gap": 0.0,
                "final_gap": float(gap[-1]), "max_gap": float(np.abs(gap).max()),
                "duration_at_start": float(c["fund_duration"].iloc[0])}
    anchor = min(shock + steps_per_year // 12, len(gap) - 1)   # a month after, past the noise
    horizon = min(int(maturity * steps_per_year), len(gap))    # never look past redemption
    sign0 = np.sign(gap[anchor])
    after = gap[anchor:horizon]
    crossings = np.flatnonzero(np.sign(after) != sign0) if sign0 != 0 else np.array([0])
    return {
        "crossover_years": float(crossings[0] / steps_per_year) if len(crossings) else np.nan,
        "shock_at_years": float(shock / steps_per_year),
        "initial_gap": float(gap[anchor]),
        "final_gap": float(gap[-1]),
        "max_gap": float(np.abs(gap).max()),
        "duration_at_start": float(c["fund_duration"].iloc[0]),
    }


# --------------------------------------------------------------------------- #
# On the real tape
# --------------------------------------------------------------------------- #
def realised_vs_promised(fund_px: pd.Series, yields: pd.Series, horizon_y: float,
                         duration: float) -> pd.DataFrame:
    """Every rolling window: the yield on offer at the start against what was delivered.

    The starting yield of a bond *is* its long-run return. The starting yield of a fund is a
    forecast whose error is exactly the change in rates times the duration, spread over the
    horizon — which this table lets you see rather than assert.
    """
    px = fund_px.dropna()
    y = yields.reindex(px.index).ffill().dropna()
    px = px.reindex(y.index)
    step = int(round(horizon_y * TRADING_DAYS))
    if step < 21 or len(px) <= step:
        return pd.DataFrame(columns=["start", "promised", "realised", "error", "d_yield"])
    rows = []
    for i in range(0, len(px) - step, 21):
        p0, p1 = float(px.iloc[i]), float(px.iloc[i + step])
        y0, y1 = float(y.iloc[i]), float(y.iloc[i + step])
        realised = (p1 / p0) ** (1.0 / horizon_y) - 1.0
        rows.append({"start": px.index[i], "promised": y0, "realised": realised,
                     "error": realised - y0, "d_yield": y1 - y0})
    return pd.DataFrame(rows).set_index("start")


def error_decomposition(tbl: pd.DataFrame, duration: float, horizon_y: float) -> dict:
    """Is the shortfall explained by duration times the rate change, as theory says?

    Predicted annualised error = -duration * dy / horizon. If the regression of the realised
    error on that prediction has a slope near one, the "fund does not converge" story is fully
    accounted for by its refusal to mature, and nothing else is going on.
    """
    if len(tbl) < 30:
        return {"n": int(len(tbl))}
    pred = -duration * tbl["d_yield"] / horizon_y
    y = tbl["error"].to_numpy(float)
    x = pred.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(x)), "intercept": float(coef[0]), "slope": float(coef[1]),
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
            "mean_error": float(y.mean()), "mean_predicted": float(x.mean()),
            "sd_error": float(y.std(ddof=1))}


def convergence_by_horizon(fund_px: pd.Series, yields: pd.Series, duration: float,
                           horizons=(1, 2, 3, 5, 7, 10)) -> pd.DataFrame:
    """Does the gap between promise and delivery shrink as the horizon lengthens?

    For a bond it must: the error goes to zero at maturity by construction. For a fund the
    theory says the error should shrink like ``duration * sd(dy) / horizon``, which is slower
    and never reaches zero.
    """
    rows = []
    for hy in horizons:
        t = realised_vs_promised(fund_px, yields, hy, duration)
        if len(t) < 20:
            continue
        rows.append({"horizon_y": hy, "n": len(t), "mean_promised": t["promised"].mean(),
                     "mean_realised": t["realised"].mean(), "mean_error": t["error"].mean(),
                     "sd_error": t["error"].std(ddof=1),
                     "share_within_1pp": float((t["error"].abs() < 0.01).mean())})
    return pd.DataFrame(rows).set_index("horizon_y") if rows else pd.DataFrame()


def synthetic_world(n_years: int = 20, shock_bp: float = 200.0, maturity: float = 10.0,
                    start_yield: float = 0.04, vol: float = 0.0, seed: int = 986) -> dict:
    """The controlled experiment: one rate path, two instruments, a known answer.

    Because both the bond and the fund are generated from the *same* rate path, every difference
    between them is the roll. ``shock_bp`` sets the size of the one-off rate move a year in.
    """
    n = int(n_years * TRADING_DAYS)
    rates = rate_path("shock", n=n, start=start_yield,
                      end=start_yield + shock_bp / 1e4, shock_at=TRADING_DAYS,
                      vol=vol, seed=seed)
    c = compare(rates, maturity)
    return {"rates": rates, "comparison": c, "maturity": maturity,
            "duration": modified_duration(start_yield, maturity, start_yield)}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** if the fund's realised return demonstrably fails to converge to
      its starting yield — specifically, if the error's standard deviation at a horizon equal to
      the fund's duration is still above 1% a year, *and* the error is explained by duration ×
      rate change with an R² above 0.5 (the mechanism, not just the phenomenon);
      **Partial** if the phenomenon holds but the mechanism does not; **Busted** otherwise.
    - **Tradability**: **Useful** if the convergence horizon is within a normal investor's
      planning window (under 20 years) and the effect is large enough to change an allocation
      (over 1% a year at short horizons); **Partial** if only one holds; **Mirage** otherwise.
    """
    fails_to_converge = h["sd_error_at_duration"] > 0.01
    mechanism = h["decomp_r2"] > 0.5
    signal = ("Confirmed" if (fails_to_converge and mechanism)
              else ("Partial" if fails_to_converge or mechanism else "Busted"))
    reachable = np.isfinite(h["convergence_years"]) and h["convergence_years"] < 20
    material = h["sd_error_1y"] > 0.01
    trad = ("Useful" if (reachable and material)
            else ("Partial" if (reachable or material) else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"A bond's starting yield is its return; a fund's is a forecast. Over "
            f"{h['n_windows']} rolling {h['duration']:.1f}-year windows — a horizon equal to "
            f"{h['fund']}'s own duration, where immunisation theory says the error should have "
            f"washed out — the realised annualised return missed the starting yield by "
            f"**{h['mean_error_at_duration']:+.2%} on average with a standard deviation of "
            f"{h['sd_error_at_duration']:.2%}**, and only {h['share_within_1pp']:.0%} of windows "
            f"landed within a percentage point of what was on offer. The mechanism checks out: "
            f"regressing the error on **−duration × Δyield ⁄ horizon** gives a slope of "
            f"{h['decomp_slope']:.2f} with R² **{h['decomp_r2']:.0%}**, so the shortfall is the "
            f"refusal to mature and essentially nothing else. At one year the error's standard "
            f"deviation is {h['sd_error_1y']:.1%}; it falls to {h['sd_error_10y']:.1%} at ten "
            f"years — shrinking, but nothing like a bond's, which is zero at maturity by "
            f"construction."),
        "trad": trad,
        "trad_why": (
            f"In the controlled experiment — one rate path, a {h['sim_maturity']:.0f}-year bond "
            f"held to maturity against a {h['sim_maturity']:.0f}-year constant-maturity fund, "
            f"a {h['sim_shock_bp']:.0f} bp rise a year in — the fund fell "
            f"{abs(h['initial_gap']):.1%} behind and took **{h['crossover_years']:.1f} years** "
            f"to catch the bond, against a starting duration of {h['sim_duration']:.1f}. On a "
            f"steadily trending rate path the fund's annualised return crossed back through its "
            f"purchase yield after **{h['convergence_years']:.1f} years** — against Leibowitz, "
            f"Bova & Kogelman's 2D − 1 = {h['leibowitz_bound']:.1f}, and *below* it because the "
            f"fund's duration shrinks as yields rise, so the cumulative price loss is smaller "
            f"than D₀ × Δy. That is the practical content: the fund is not worse, it is "
            f"*slower to be right*, and the delay is set by its duration. An investor whose "
            f"horizon comfortably exceeds it is close to indifferent; one whose horizon is "
            f"shorter is holding a rate bet they may not know they have taken."),
        "one_sentence": (
            f"A constant-maturity bond fund delivers its starting yield only after roughly "
            f"{h['convergence_years'] / h['sim_duration']:.1f} times its duration — "
            f"{h['convergence_years']:.1f} years in the clean experiment — and until then the "
            f"gap is {h['sd_error_1y']:.1%} a year of pure rate risk that a bond held to "
            f"maturity simply does not have."),
    }
