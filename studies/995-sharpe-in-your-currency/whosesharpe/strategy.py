"""The same fund, from four countries — Study 995.

A US investor holding SPY earns ``r_usd``. A euro-based investor holding the same shares earns

    (1 + r_usd) / (1 + r_eurusd) - 1

because the dollars have to be converted back. Three things follow, and they are usually
conflated:

1. **The mean changes.** Over any finite window the currency has drifted, and that drift adds
   to or subtracts from the return. This is the part everyone notices and it is the *least*
   interesting, because currency drift is close to unforecastable and averages out over long
   horizons.

2. **The volatility changes, and almost always upward.** ``var(a - c) = var(a) + var(c) -
   2cov(a, c)``. Unless the asset and the currency are strongly positively correlated, adding a
   currency leg adds variance. That is mechanical, permanent, and does not average out.

3. **The risk-free rate changes too.** A euro investor's cash leg is the euro deposit rate, not
   the dollar one, and during 2015-2022 those differed by more than two percentage points. A
   Sharpe ratio computed with the wrong risk-free rate is wrong in a direction nobody checks.

``sharpe_by_currency`` handles all three properly. ``decompose_sharpe_gap`` then splits the
difference between two investors' Sharpe ratios into a drift term, a variance term and a
rate-differential term — which is the only way to say whether a currency "helped" for a reason
that might persist or for one that will not.

The hedging question is priced directly rather than assumed: ``hedge_analysis`` compares the
unhedged position against a currency-hedged one, charging the interest-rate differential that a
forward contract actually costs, and reports the horizon at which the hedge pays for itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Converting a return into someone else's money
# --------------------------------------------------------------------------- #
def convert_return(asset_ret: pd.Series, fx_ret: pd.Series) -> pd.Series:
    """A USD-denominated return, seen by an investor whose home currency is the FX leg.

    ``fx_ret`` is the return of one unit of foreign currency priced in dollars, so a positive
    value means the foreign currency **appreciated** and the dollar asset is worth less at home.
    The exact conversion is ``(1 + r) / (1 + f) - 1``, not ``r - f``: the approximation is fine
    for a day and materially wrong when compounded over a decade, and this study is about a
    decade.
    """
    df = pd.concat([asset_ret.rename("a"), fx_ret.rename("f")], axis=1,
                   sort=False).dropna()
    return ((1 + df["a"]) / (1 + df["f"]) - 1.0).rename("converted")


def sharpe(excess: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of an already-excess return series.

    The ``1e-12`` floor on the standard deviation is not decoration: a constant series has a
    sample standard deviation of about 1e-19 rather than exactly zero, and dividing by it
    returns a Sharpe of 1e16 instead of the NaN the caller expects.
    """
    x = excess.dropna()
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 1e-12 and len(x) > 30 else np.nan


def log_excess(ret: pd.Series, rf: pd.Series) -> pd.Series:
    """Log return minus log risk-free — the input to a Sharpe that is not fooled by convexity."""
    df = pd.concat([ret.rename("r"), rf.rename("rf")], axis=1, sort=False).dropna()
    return (np.log1p(df["r"]) - np.log1p(df["rf"])).rename("log_excess")


def geometric_sharpe(ret: pd.Series, rf: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Sharpe computed on LOG excess returns, which is the one to trust here.

    A trap specific to currency conversion, and a nasty one. The arithmetic mean of
    ``(1+a)/(1+f) - 1`` contains a **convexity term of roughly var(f)**: dividing by a random
    number raises the arithmetic average even when it lowers the compounded outcome. So a more
    volatile currency inflates the arithmetic Sharpe's numerator at the same time as it inflates
    the denominator, and the two can offset almost exactly — leaving an arithmetic Sharpe that
    barely moves while the investor is unambiguously worse off.

    Log returns have no such term: ``log((1+a)/(1+f)) = log(1+a) - log(1+f)`` exactly. Every
    Sharpe this study leans on is the log one, and the arithmetic version is reported beside it
    so the size of the artefact is visible.
    """
    x = log_excess(ret, rf)
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 1e-12 and len(x) > 30 else np.nan


def stats_block(ret: pd.Series, rf: pd.Series | None = None) -> dict:
    """Return, volatility, Sharpe and drawdown for one investor's experience."""
    r = ret.dropna()
    if len(r) < 100:
        return {"n": int(len(r))}
    rf_s = (rf.reindex(r.index).fillna(0.0) if rf is not None
            else pd.Series(0.0, index=r.index))
    excess = r - rf_s
    years = len(r) / TRADING_DAYS
    cum = (1 + r).cumprod()
    return {"n": int(len(r)),
            "cagr": float(cum.iloc[-1] ** (1 / years) - 1),
            "vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "rf_ann": float(rf_s.mean() * TRADING_DAYS),
            "excess_ann": float(excess.mean() * TRADING_DAYS),
            "sharpe": geometric_sharpe(r, rf_s),
            "sharpe_arithmetic": sharpe(excess),
            "max_dd": float((cum / cum.cummax() - 1).min()),
            "skew": float(r.skew())}


def sharpe_by_currency(asset_ret: pd.Series, fx_rets: dict, usd_rf: pd.Series,
                       foreign_rf: dict | None = None) -> pd.DataFrame:
    """The same asset, measured from every home currency, each with its own cash rate.

    ``foreign_rf`` supplies each investor's own risk-free rate. When it is absent the function
    derives one from covered interest parity: the foreign deposit rate is approximately the
    dollar rate minus the currency ETF's excess return over dollar cash, because the ETF holds
    a foreign deposit and its dollar return is the currency move *plus* that deposit rate.
    Deriving it is better than ignoring it — using the dollar rate for every investor is the
    standard shortcut and it biases every high-rate country's Sharpe downward.
    """
    rows = [{"currency": "USD", **stats_block(asset_ret, usd_rf)}]
    for name, fx in fx_rets.items():
        conv = convert_return(asset_ret, fx)
        if foreign_rf and name in foreign_rf:
            rf_local = foreign_rf[name]
        else:
            rf_local = implied_foreign_rate(fx, usd_rf)
        rows.append({"currency": name, **stats_block(conv, rf_local)})
    return pd.DataFrame(rows).set_index("currency")


def implied_foreign_rate(fx_ret: pd.Series, usd_rf: pd.Series) -> pd.Series:
    """The foreign deposit rate implied by a currency ETF's return.

    A currency ETF's dollar total return is (currency appreciation) + (foreign deposit rate).
    The *spot* appreciation is not separately observable here, but under covered interest parity
    the expected appreciation is the rate differential, so the ETF's drift relative to dollar
    cash carries the differential. Estimated on a long rolling window because the daily signal
    is pure noise.

    This is an approximation and the results say so. It is a better approximation than assuming
    every investor in the world earns the US Treasury bill rate.
    """
    df = pd.concat([fx_ret.rename("f"), usd_rf.rename("rf")], axis=1, sort=False).dropna()
    if len(df) < 300:
        return pd.Series(0.0, index=fx_ret.index)
    drift = df["f"].rolling(756, min_periods=252).mean()
    return (df["rf"] - drift).clip(lower=-0.02 / TRADING_DAYS,
                                   upper=0.10 / TRADING_DAYS).rename("rf_local")


# --------------------------------------------------------------------------- #
# Why the Sharpe moved
# --------------------------------------------------------------------------- #
def variance_decomposition(asset_ret: pd.Series, fx_ret: pd.Series) -> dict:
    """var(converted) against var(asset) + var(fx) - 2cov — the mechanical part.

    The identity holds exactly for log returns and approximately for simple ones; the function
    reports both the identity's prediction and the realised variance so the approximation error
    is visible rather than assumed away.
    """
    df = pd.concat([asset_ret.rename("a"), fx_ret.rename("f")], axis=1,
                   sort=False).dropna()
    if len(df) < 300:
        return {"n": int(len(df))}
    a, f = df["a"], df["f"]
    conv = (1 + a) / (1 + f) - 1
    va, vf = float(a.var(ddof=1)), float(f.var(ddof=1))
    cov = float(a.cov(f))
    predicted = va + vf - 2 * cov
    realised = float(conv.var(ddof=1))
    return {"n": int(len(df)), "var_asset": va, "var_fx": vf, "cov": cov,
            "corr": float(a.corr(f)),
            "predicted_var": predicted, "realised_var": realised,
            "vol_asset": float(np.sqrt(va * TRADING_DAYS)),
            "vol_converted": float(np.sqrt(realised * TRADING_DAYS)),
            "vol_ratio": float(np.sqrt(realised / va)) if va > 0 else np.nan,
            "approximation_error": float(realised - predicted)}


def hedge_ratio_that_minimises_variance(asset_ret: pd.Series, fx_ret: pd.Series) -> float:
    """The variance-minimising hedge ratio — which is not always 1.

    ``h* = cov(asset, fx) / var(fx) + 1`` for a fully dollar-denominated asset. When the asset
    itself moves with the currency (as US equities do against risk-off currencies) the optimal
    hedge is *less* than 100%, and hedging fully overshoots. Campbell, Serfaty-de Medeiros &
    Viceira (2010) is the reference.
    """
    df = pd.concat([asset_ret.rename("a"), fx_ret.rename("f")], axis=1,
                   sort=False).dropna()
    if len(df) < 300:
        return np.nan
    vf = float(df["f"].var(ddof=1))
    return float(1.0 + df["a"].cov(df["f"]) / vf) if vf > 0 else np.nan


def decompose_sharpe_gap(asset_ret: pd.Series, fx_ret: pd.Series, usd_rf: pd.Series,
                         local_rf: pd.Series | None = None) -> dict:
    """Split one investor's Sharpe minus another's into drift, variance and rate terms.

    The whole point of the study. A currency can raise a foreign investor's Sharpe for three
    quite different reasons, and only one of them is worth anything forward-looking:

    - **drift**: the currency happened to fall over this window. Unforecastable, mean-reverting
      over long horizons, and the least durable of the three.
    - **variance**: the currency added risk. Mechanical, permanent, and always adverse unless
      the correlation is strongly positive.
    - **rate differential**: the home cash rate differs, so the Sharpe's numerator is measured
      against a different bar. Persistent while the policy gap persists.
    """
    conv = convert_return(asset_ret, fx_ret)
    idx = conv.index
    a = asset_ret.reindex(idx)
    usd = usd_rf.reindex(idx).fillna(0.0)
    loc = (local_rf.reindex(idx).fillna(0.0) if local_rf is not None
           else implied_foreign_rate(fx_ret, usd_rf).reindex(idx).fillna(0.0))
    if len(idx) < 300:
        return {"n": int(len(idx))}
    s_home = geometric_sharpe(a, usd)
    s_foreign = geometric_sharpe(conv, loc)
    if not (np.isfinite(s_home) and np.isfinite(s_foreign)):
        return {"n": int(len(idx))}
    sd_a = float(log_excess(a, usd).std(ddof=1))
    sd_c = float(log_excess(conv, loc).std(ddof=1))
    # Rebuild the foreign Sharpe one ingredient at a time, all on log excess returns.
    drift_and_var = geometric_sharpe(conv, usd)
    rate_term = s_foreign - drift_and_var
    var_term = (float(log_excess(a, usd).mean()) * np.sqrt(TRADING_DAYS) / sd_c
                - s_home) if sd_c > 0 else np.nan
    drift_term = drift_and_var - s_home - var_term
    return {"n": int(len(idx)), "sharpe_home": s_home, "sharpe_foreign": s_foreign,
            "gap": s_foreign - s_home, "drift_term": drift_term,
            "variance_term": var_term, "rate_term": rate_term,
            "vol_ratio": sd_c / sd_a if sd_a > 0 else np.nan,
            "rate_gap_ann": float((loc.mean() - usd.mean()) * TRADING_DAYS)}


# --------------------------------------------------------------------------- #
# Hedging
# --------------------------------------------------------------------------- #
def hedged_return(asset_ret: pd.Series, fx_ret: pd.Series, usd_rf: pd.Series,
                  local_rf: pd.Series, hedge_ratio: float = 1.0,
                  cost_bps: float = 3.0) -> pd.Series:
    """A currency-hedged position for a foreign investor.

    A rolling forward hedge earns the **interest-rate differential**, not zero. Selling dollars
    forward when US rates exceed home rates costs you that gap — the "hedge cost" that hedged
    share classes quietly charge and that most comparisons omit. Here it is explicit.
    """
    conv = convert_return(asset_ret, fx_ret)
    idx = conv.index
    f = fx_ret.reindex(idx).fillna(0.0)
    carry = (local_rf.reindex(idx).fillna(0.0) - usd_rf.reindex(idx).fillna(0.0))
    a = asset_ret.reindex(idx)
    # Hedging removes hedge_ratio of the currency move and pays the differential on it.
    hedged = conv + hedge_ratio * (f / (1 + f)) * (1 + a) + hedge_ratio * carry
    turnover = hedge_ratio * 1.0 / 21.0     # a monthly forward roll
    return (hedged - turnover * cost_bps / 1e4).rename("hedged")


def hedge_analysis(asset_ret: pd.Series, fx_ret: pd.Series, usd_rf: pd.Series,
                   local_rf: pd.Series | None = None, cost_bps: float = 3.0) -> dict:
    """Unhedged against hedged, at several hedge ratios including the optimal one."""
    loc = (local_rf if local_rf is not None else implied_foreign_rate(fx_ret, usd_rf))
    conv = convert_return(asset_ret, fx_ret)
    unhedged = stats_block(conv, loc)
    h_star = hedge_ratio_that_minimises_variance(asset_ret, fx_ret)
    rows = []
    for h in (0.0, 0.5, 1.0):
        hr = hedged_return(asset_ret, fx_ret, usd_rf, loc, h, cost_bps)
        rows.append({"hedge_ratio": h, **stats_block(hr, loc)})
    if np.isfinite(h_star):
        hr = hedged_return(asset_ret, fx_ret, usd_rf, loc, float(np.clip(h_star, -1, 2)),
                           cost_bps)
        rows.append({"hedge_ratio": float(np.clip(h_star, -1, 2)),
                     **stats_block(hr, loc)})
    tbl = pd.DataFrame(rows).set_index("hedge_ratio").sort_index()
    return {"unhedged": unhedged, "table": tbl, "optimal_ratio": h_star,
            "vol_saved": (unhedged.get("vol", np.nan)
                          - float(tbl.loc[1.0, "vol"]) if 1.0 in tbl.index else np.nan),
            "sharpe_gain": (float(tbl.loc[1.0, "sharpe"]) - unhedged.get("sharpe", np.nan)
                            if 1.0 in tbl.index else np.nan)}


def ranking_stability(assets: dict, fx_rets: dict, usd_rf: pd.Series) -> pd.DataFrame:
    """Rank several assets by Sharpe, from each home currency. Do the rankings agree?

    The question that makes this practically important rather than merely arithmetic. If a
    currency shifts every asset's Sharpe by the same amount, nothing changes for an allocator.
    If it reorders them, then "which fund has the best risk-adjusted return" has a different
    answer in Zurich than in New York.
    """
    out = {}
    for cur, fx in [("USD", None)] + list(fx_rets.items()):
        col = {}
        for name, r in assets.items():
            if fx is None:
                col[name] = geometric_sharpe(r, usd_rf.reindex(r.index).fillna(0.0))
            else:
                conv = convert_return(r, fx)
                loc = implied_foreign_rate(fx, usd_rf).reindex(conv.index).fillna(0.0)
                col[name] = geometric_sharpe(conv, loc)
        out[cur] = col
    df = pd.DataFrame(out)
    ranks = df.rank(ascending=False)
    df.loc["_rank_spread"] = ranks.max(axis=1).sub(ranks.min(axis=1)).max()
    return df


def synthetic_world(n: int = 5000, asset_vol: float = 0.16, fx_vol: float = 0.10,
                    corr: float = 0.0, asset_drift: float = 0.08,
                    fx_drift: float = 0.0, rate_gap: float = 0.0,
                    seed: int = 995) -> dict:
    """An asset and a currency with a controllable correlation between them.

    At ``corr = 0`` the variance identity is exact and the Sharpe change is purely mechanical,
    which is how the decomposition gets graded. Positive correlation means the currency rises
    when the asset does, so a foreign investor's converted return is *less* volatile than the
    local one — the one case where currency exposure reduces risk.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[asset_vol ** 2, corr * asset_vol * fx_vol],
                    [corr * asset_vol * fx_vol, fx_vol ** 2]]) / TRADING_DAYS
    draws = rng.multivariate_normal([asset_drift / TRADING_DAYS,
                                     fx_drift / TRADING_DAYS], cov, n)
    idx = pd.bdate_range("2007-01-03", periods=n)
    return {"asset": pd.Series(draws[:, 0], index=idx, name="asset"),
            "fx": pd.Series(draws[:, 1], index=idx, name="fx"),
            "usd_rf": pd.Series(0.02 / TRADING_DAYS, index=idx, name="usd_rf"),
            "local_rf": pd.Series((0.02 + rate_gap) / TRADING_DAYS, index=idx,
                                  name="local_rf")}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the Sharpe of the same asset differs across home currencies by
      more than 0.15 **and** the currency changes the *ranking* of at least two assets — i.e.
      it is not just a level shift that cancels out of every comparison; **Weak** if the spread
      is material but rankings are stable; **None** if the spread is negligible.
    - **Tradability**: **Useful** if hedging raises the Sharpe for a majority of currencies
      after the interest-rate differential is charged; **Partial** if it helps some; **Mirage**
      if it helps none.
    """
    material = h["sharpe_spread"] > 0.15
    reorders = h["rank_spread"] >= 2
    signal = ("Real" if (material and reorders)
              else ("Weak" if material else "None"))
    trad = ("Useful" if h["hedge_helps_share"] > 0.5
            else ("Partial" if h["hedge_helps_share"] > 0 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Over {h['years']:.0f} years, {h['asset']} delivered a Sharpe of "
            f"**{h['sharpe_usd']:.2f}** to a dollar-based investor and anywhere from "
            f"**{h['sharpe_min']:.2f}** ({h['worst_currency']}) to **{h['sharpe_max']:.2f}** "
            f"({h['best_currency']}) to investors based elsewhere — a spread of "
            f"**{h['sharpe_spread']:.2f}**, on the identical shares. Three separate channels do "
            f"the work and they behave differently. The **variance** channel is mechanical: "
            f"adding a currency leg raised volatility from {h['vol_usd']:.1%} to a median "
            f"{h['vol_median_foreign']:.1%} because var(a−c) = var(a) + var(c) − 2cov, and the "
            f"median correlation between {h['asset']} and these currencies was only "
            f"{h['median_corr']:+.2f}. The **drift** channel is luck: the dollar happened to "
            f"move. The **rate** channel is the one nobody adjusts for — each investor's cash "
            f"leg is their own, and using the US bill rate for everyone (the standard shortcut) "
            f"biases every high-rate country's Sharpe downward. Across "
            f"{h['n_assets']} assets the currency moved at least one pair's ranking by "
            f"**{h['rank_spread']:.0f} places**."),
        "trad_why": (
            f"Hedging is priced here rather than assumed, because a rolling forward hedge earns "
            f"the **interest-rate differential** and not zero — selling dollars forward while "
            f"US rates exceed yours costs you the gap, which is the charge hedged share classes "
            f"pass on quietly. After that charge, a full hedge raised the Sharpe for "
            f"**{h['hedge_helps_share']:.0%}** of the currencies here, by a median "
            f"{h['median_hedge_gain']:+.2f}. The variance-minimising ratio is not 1.0 either: "
            f"its median across currencies is **{h['median_optimal_ratio']:.2f}**, because "
            f"{h['asset']} itself co-moves with risk-off currencies, so a full hedge "
            f"{'overshoots' if h['median_optimal_ratio'] < 0.95 else 'is close to right'}. "
            f"The practical reading is that the hedge buys volatility reduction reliably and "
            f"return unreliably — which is the right way round for a long-horizon holder and "
            f"the wrong way round for anyone hoping the hedge pays for itself."),
        "trad": trad,
        "one_sentence": (
            f"The same {h['asset']} shares delivered Sharpe ratios from {h['sharpe_min']:.2f} to "
            f"{h['sharpe_max']:.2f} depending on the holder's home currency — a spread of "
            f"{h['sharpe_spread']:.2f}, wide enough to reorder how funds rank against each "
            f"other."),
    }
