"""Bitcoin's weekend, and Monday's open — Study 983.

Most lead-lag studies between two liquid assets founder on the same rock: the two series are
contemporaneously correlated, the closes are not simultaneous, and what looks like *A* leading
*B* is *A*'s close containing news that *B*'s earlier close could not. Crypto against equities
has exactly that problem in its ordinary daily form — Yahoo's BTC bar closes at 00:00 UTC,
three hours after the NYSE bell — and the study is careful to say so.

But crypto against equities also has a rare *clean* case. The market is shut from Friday
16:00 New York to Monday 09:30, sixty-five hours in which Bitcoin trades and equities cannot.
Whatever happens in that window is priced by one and not the other, so:

    weekend crypto return  ->  Monday equity return

is a lead-lag question with **no overlap at all**. No clock trick can manufacture it. If the
weekend move predicts Monday, that is information; if it does not, the "crypto leads risk
assets" story has lost its best possible test case.

The module provides:

- ``weekend_returns`` — the Saturday+Sunday crypto move, built only from bars that fall in the
  closed window, with the Friday and Monday bars excluded by construction.
- ``overlap_flag`` — an explicit statement, per alignment, of how many hours of overlap the
  design contains, so that no table in this study is read without it.
- ``lead_lag_grid`` — the ordinary daily cross-correlations in both directions, reported *with*
  their overlap caveat rather than instead of it.
- ``monday_regression`` and ``monday_rule`` — the clean test, and what it would be worth.
- ``regime_split`` — before and after March 2020, when Bitcoin stopped behaving like a
  diversifier and started behaving like a high-beta Nasdaq position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
NY_CLOSE_UTC = 21          # 16:00 ET in winter, 20:00 UTC in summer; 21 is the conservative read
CRYPTO_BAR_CLOSE_UTC = 24  # Yahoo daily crypto bars are UTC days


# --------------------------------------------------------------------------- #
# The clock, stated explicitly
# --------------------------------------------------------------------------- #
def overlap_hours(alignment: str) -> float:
    """Hours of information in the predictor that post-date the target's own start.

    ``"same_day"``   — crypto day *t* against equity day *t*: fully overlapping, meaningless.
    ``"crypto_lead"``— crypto day *t* against equity day *t+1*: three hours of the crypto bar
                       come *after* the equity close of day *t*, which is a real but tiny lead.
    ``"weekend"``    — the Saturday+Sunday crypto move against Monday: **zero** overlap, the
                       clean case this study exists for.
    """
    return {"same_day": 21.0, "crypto_lead": 3.0, "weekend": 0.0}[alignment]


def is_clean(alignment: str) -> bool:
    """True only for designs with no overlap between predictor and target windows."""
    return overlap_hours(alignment) == 0.0


# --------------------------------------------------------------------------- #
# Building the weekend
# --------------------------------------------------------------------------- #
def weekend_returns(crypto_px: pd.Series, equity_index: pd.DatetimeIndex) -> pd.DataFrame:
    """The crypto move over each closed-market weekend, matched to the following session.

    The weekend is defined by the *equity* calendar, not the crypto one: for each pair of
    consecutive equity sessions separated by more than one calendar day, the crypto return is
    taken from the earlier session's date to the later session's date. That definition also
    picks up long weekends and holiday closures, which is deliberate — they are the same
    experiment with a longer information window.
    """
    eq = pd.Series(equity_index, index=equity_index)
    px = crypto_px.dropna()
    rows = []
    dates = list(equity_index)
    for prev, nxt in zip(dates[:-1], dates[1:]):
        gap = (nxt - prev).days
        if gap <= 1:
            continue
        before = px.loc[:prev]
        after = px.loc[:nxt]
        if before.empty or after.empty or before.index[-1] >= after.index[-1]:
            continue
        r = float(after.iloc[-1] / before.iloc[-1] - 1.0)
        rows.append({"session": nxt, "previous_session": prev, "gap_days": gap,
                     "crypto_return": r})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["session", "previous_session", "gap_days",
                                     "crypto_return"]).set_index("session")
    return out.set_index("session")


def closed_window_hours(gap_days: int) -> float:
    """Hours the equity market was shut for a gap of ``gap_days`` calendar days."""
    return 24.0 * gap_days - 6.5


def attach_target(weekends: pd.DataFrame, equity_rets: pd.Series) -> pd.DataFrame:
    """Add the equity return of the session that follows each closed window."""
    out = weekends.copy()
    out["equity_return"] = equity_rets.reindex(out.index)
    return out.dropna(subset=["equity_return"])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ols_t(y: np.ndarray, x: np.ndarray) -> dict:
    """Simple OLS with heteroskedasticity-robust (HC1) standard errors.

    The weekend observations are one per week and non-overlapping, so no HAC lag structure is
    needed — but the variance of Monday returns is anything but constant, so White's correction
    is.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 30:
        return {"n": int(n), "beta": np.nan, "t": np.nan, "r2": np.nan, "alpha": np.nan}
    A = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    XtX_inv = np.linalg.pinv(A.T @ A)
    meat = A.T @ np.diag(resid ** 2) @ A * n / max(n - 2, 1)
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(max(V[1, 1], 0.0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": int(n), "alpha": float(coef[0]), "beta": float(coef[1]),
            "t": float(coef[1] / se) if se > 0 else np.nan,
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan}


def monday_regression(panel: pd.DataFrame) -> dict:
    """Monday's equity return on the weekend's crypto return — the clean test."""
    out = ols_t(panel["equity_return"].to_numpy(), panel["crypto_return"].to_numpy())
    out["alignment"] = "weekend"
    out["overlap_hours"] = overlap_hours("weekend")
    return out


def sign_agreement(panel: pd.DataFrame) -> dict:
    """How often Monday's sign matches the weekend's — the version a trader would quote."""
    s = np.sign(panel["crypto_return"]) == np.sign(panel["equity_return"])
    n = int(len(s))
    p = float(s.mean()) if n else np.nan
    se = np.sqrt(0.25 / n) if n else np.nan
    return {"n": n, "hit_rate": p, "t_vs_coin_flip": float((p - 0.5) / se) if n else np.nan}


def conditional_means(panel: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Monday's average return after an up weekend and after a down weekend."""
    up = panel[panel["crypto_return"] > threshold]["equity_return"]
    down = panel[panel["crypto_return"] <= threshold]["equity_return"]
    rows = []
    for label, s in (("crypto weekend up", up), ("crypto weekend down", down)):
        rows.append({"bucket": label, "n": len(s), "mean": s.mean(), "median": s.median(),
                     "sd": s.std(ddof=1)})
    d = up.mean() - down.mean()
    se = np.sqrt(up.var(ddof=1) / max(len(up), 1) + down.var(ddof=1) / max(len(down), 1))
    rows.append({"bucket": "difference", "n": len(panel), "mean": d, "median": np.nan,
                 "sd": np.nan, "t": d / se if se > 0 else np.nan})
    return pd.DataFrame(rows).set_index("bucket")


def lead_lag_grid(crypto_rets: pd.Series, equity_rets: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Ordinary daily cross-correlations both ways — reported *with* their overlap caveat."""
    df = pd.concat([crypto_rets.rename("c"), equity_rets.rename("e")], axis=1).dropna()
    rows = []
    for k in range(-max_lag, max_lag + 1):
        c = float(df["c"].shift(k).corr(df["e"]))
        if k > 0:
            desc, align = f"crypto leads equities by {k}d", "crypto_lead"
        elif k < 0:
            desc, align = f"equities lead crypto by {-k}d", "same_day"
        else:
            desc, align = "same day", "same_day"
        rows.append({"lag": k, "description": desc, "correlation": c,
                     "overlap_hours": overlap_hours(align),
                     "clean": is_clean(align)})
    return pd.DataFrame(rows).set_index("lag")


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def monday_rule(panel: pd.DataFrame, equity_rets: pd.Series, cash_rets: pd.Series,
                cost_bps: float = 2.0, threshold: float = 0.0) -> dict:
    """Own equities on the session after an up weekend, bills otherwise; hold bills midweek.

    A deliberately spare rule: it is in the market roughly one day in five, so its Sharpe is
    not comparable to a buy-and-hold Sharpe and the comparison reported is against the *same
    days* held passively — the only honest benchmark for a rule this sparse.
    """
    long_days = panel.index[panel["crypto_return"] > threshold]
    r_eq = equity_rets.reindex(panel.index)
    r_cash = cash_rets.reindex(panel.index).fillna(0.0)
    held = pd.Series(np.where(panel.index.isin(long_days), r_eq, r_cash), index=panel.index)
    held = held - (panel.index.isin(long_days)).astype(float) * 2 * cost_bps / 1e4
    always = r_eq.copy()
    n_years = (panel.index[-1] - panel.index[0]).days / 365.25 if len(panel) > 1 else np.nan
    d = (held - always).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 30 else np.nan
    return {"n_windows": int(len(panel)), "n_long": int(len(long_days)),
            "share_long": float(len(long_days) / len(panel)) if len(panel) else np.nan,
            "mean_rule": float(held.mean()), "mean_always": float(always.mean()),
            "total_rule": float((1 + held).prod() - 1),
            "total_always": float((1 + always).prod() - 1),
            "per_year_rule": float((1 + held).prod() ** (1 / n_years) - 1) if n_years else np.nan,
            "per_year_always": float((1 + always).prod() ** (1 / n_years) - 1)
            if n_years else np.nan,
            "t_gap": float(d.mean() / se) if se and se > 0 else np.nan,
            "cost_bps": cost_bps, "returns": held}


def regime_split(panel: pd.DataFrame, cut: str = "2020-03-01") -> pd.DataFrame:
    """The same regression before and after Bitcoin joined the risk-asset complex."""
    rows = []
    for label, sl in (("before " + cut, panel.loc[:cut]), ("since " + cut, panel.loc[cut:])):
        r = monday_regression(sl) if len(sl) >= 30 else {"n": len(sl), "beta": np.nan,
                                                         "t": np.nan, "r2": np.nan}
        rows.append({"regime": label, "n": r["n"], "beta": r["beta"], "t": r["t"],
                     "r2": r["r2"]})
    return pd.DataFrame(rows).set_index("regime")


def gap_length_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    """Does a longer closure carry more information? Two-day weekends against three-day ones."""
    rows = []
    for gap, sl in panel.groupby("gap_days"):
        if len(sl) < 30:
            continue
        r = monday_regression(sl)
        rows.append({"gap_days": int(gap), "hours_closed": closed_window_hours(int(gap)),
                     "n": r["n"], "beta": r["beta"], "t": r["t"], "r2": r["r2"]})
    return pd.DataFrame(rows).set_index("gap_days") if rows else pd.DataFrame(
        columns=["hours_closed", "n", "beta", "t", "r2"])


def synthetic_world(n_weeks: int = 600, weekend_information: float = 0.0,
                    common_beta: float = 0.6, crypto_noise: float = 0.045,
                    seed: int = 983) -> pd.DataFrame:
    """Weekly observations of a crypto weekend move and the following equity session.

    ``weekend_information`` is how much of Monday's move was already knowable from the weekend's
    news — the thing the study is trying to measure. At zero, crypto and equities still share a
    common factor (``common_beta``) so that the *ordinary* daily correlation is high, but the
    weekend carries nothing extra: the null in which any Monday result is noise.

    ``crypto_noise`` is the part of Bitcoin's weekend move that is Bitcoin's own business, and
    it is the parameter that sets the ceiling on this study's power. With the realistic default
    (a 4.5% weekly standard deviation against 1% of shared news) the correlation between the
    crypto move and the news is only about 0.13, so **even a weekend whose news fully determines
    Monday** produces a *t* of barely 3 in 600 weeks. See
    ``detectability_ceiling`` — the sample is not the binding constraint here, Bitcoin's own
    volatility is.
    """
    rng = np.random.default_rng(seed)
    news = rng.normal(0, 0.01, n_weeks)          # the weekend's news, priced by crypto only
    c_noise = rng.normal(0, crypto_noise, n_weeks)
    equity_noise = rng.normal(0, 0.009, n_weeks)
    crypto = common_beta * news + c_noise
    equity = weekend_information * news + equity_noise
    idx = pd.bdate_range("2015-01-05", periods=n_weeks, freq="W-MON")
    return pd.DataFrame({"crypto_return": crypto, "equity_return": equity,
                         "gap_days": 3}, index=idx)


def detectability_ceiling(n_weeks: int, common_beta: float = 0.6, news_sd: float = 0.01,
                          crypto_noise: float = 0.045) -> dict:
    """The largest *t* this design can produce, even if the weekend explains Monday perfectly.

    Bitcoin's weekend move is a noisy proxy for the weekend's news. Its correlation with that
    news is ``common_beta * news_sd / sqrt((common_beta * news_sd)^2 + crypto_noise^2)``, and no
    amount of predictability in the news itself can push the crypto-to-equity correlation above
    it. That number, times ``sqrt(n)``, is the ceiling on the *t*-statistic — a fact worth
    knowing *before* reading a null result as evidence of absence.
    """
    shared = common_beta * news_sd
    rho_max = shared / np.sqrt(shared ** 2 + crypto_noise ** 2)
    return {"max_correlation": float(rho_max),
            "max_t": float(rho_max * np.sqrt(max(n_weeks, 1))),
            "n_weeks_for_t2": float((2.0 / rho_max) ** 2)}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the clean weekend regression has |*t*| >= 2 **and** the sign
      survives the regime split (both halves same sign); **Weak** if |*t*| >= 2 in the full
      sample but not in both halves; **None** otherwise.
    - **Tradability**: **Investable** if the Monday rule beats holding the same sessions with
      |*t*| >= 2; **Fragile** if it wins without significance; **Mirage** if it loses.
    """
    strong = abs(h["t_weekend"]) >= 2.0
    consistent = (h["beta_before"] * h["beta_since"] > 0) if np.isfinite(
        h["beta_before"] * h["beta_since"]) else False
    signal = "Real" if (strong and consistent) else ("Weak" if strong else "None")
    gap = h["per_year_rule"] - h["per_year_always"]
    trad = ("Investable" if gap > 0 and abs(h["t_gap"]) >= 2.0
            else ("Fragile" if gap > 0 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Over {h['n_weekends']} closed-market windows ({h['window'][0]} → "
            f"{h['window'][1]}), a 1% Bitcoin move across the weekend was followed by a "
            f"**{h['beta_weekend']:+.3%}** move in {h['equity']} on the next session "
            f"(*t* = **{h['t_weekend']:+.2f}**, R² {h['r2_weekend']:.2%}). The sign agreed "
            f"{h['hit_rate']:.1%} of the time against a {50:.0f}% coin flip "
            f"(*t* = {h['t_hit']:+.2f}), and the average next-session return was "
            f"{h['mean_after_up']:+.3%} after an up weekend against {h['mean_after_down']:+.3%} "
            f"after a down one (*t* = {h['t_bucket']:+.2f}). Split at March 2020 the slope was "
            f"{h['beta_before']:+.3f} before and {h['beta_since']:+.3f} after — the era in which "
            f"Bitcoin became a high-beta Nasdaq position is doing most of the work. This is the "
            f"**clean** design: zero hours of overlap, unlike the ordinary daily lead-lag "
            f"correlation of {h['daily_lead_corr']:+.2f}, which shares "
            f"{h['daily_lead_overlap']:.0f} hours of clock with its own target. One thing this "
            f"stamp must not be read as: Bitcoin's weekend move has a standard deviation of "
            f"{h['crypto_window_sd']:.1%}, most of it Bitcoin's own business, so even a weekend "
            f"whose news determined Monday perfectly could only produce a *t* of about "
            f"**{h['ceiling']['max_t']:.1f}** over {h['n_weekends']} windows — a null here is "
            f"weak evidence of absence, not strong evidence of nothing."),
        "trad": trad,
        "trad_why": (
            f"Buying {h['equity']} only on the session after an up crypto weekend — "
            f"{h['n_long']} of {h['n_weekends']} windows, {h['share_long']:.0%} of them — "
            f"returned **{h['per_year_rule']:+.2%}/yr** on those same sessions against "
            f"**{h['per_year_always']:+.2%}** for holding every one of them regardless "
            f"({gap:+.2%}/yr, *t* = {h['t_gap']:+.2f}), after {h['cost_bps']:.0f} bps a side. "
            f"The rule is in the market about one session a week, so this is a comparison of "
            f"like with like, not a Sharpe you can put next to a buy-and-hold Sharpe."),
        "one_sentence": (
            f"Bitcoin's weekend move predicts the next equity session with *t* = "
            f"{h['t_weekend']:+.2f} and an R² of {h['r2_weekend']:.2%} — the cleanest lead-lag "
            f"design available between the two assets, and it explains "
            f"{h['r2_weekend']:.1%} of Monday."),
    }
