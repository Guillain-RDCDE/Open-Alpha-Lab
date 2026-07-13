"""Strategy + inference for Study 762 — Vegas-Gaming-Win.

The believers' rule (top-down gaming-sector timing): **rising Las Vegas Strip gross gaming
revenue (GGR) momentum precedes a run in the casino stocks.** Operationalised on the monthly
Strip-GGR tape:

    Let ``g`` = the Strip GGR level (US$ millions). Because GGR is strongly seasonal (March
    and summer peaks, February troughs) and had a total COVID shutdown, we measure momentum
    on the **trailing-12-month sum** ``T_t = sum(g_{t-11..t})`` — a deseasonalised run-rate —
    and define ``m_t = T_t / T_{t-k} - 1`` its k-month momentum (default k = 3). GGR is
    "RISING" at month t when ``m_t > thresh`` (default 0); the believers say the forward
    casino-basket return is then ELEVATED relative to the unconditional mean.

We test it by splitting forward H-month casino-basket returns into a RISING-GGR set and a
FALLING-GGR set and comparing each to the unconditional mean, with:

  * a **Welch two-sample t** of the RISING-set forward mean against the unconditional
    forward mean (a real leading signal => RISING mean is *significantly positive-excess*);
  * a **placebo / randomization null** — draw the same number of random months and ask how
    often a random draw's mean forward return is at least as *high* as the RISING set (the
    honest small-sample test for a bullish leading signal);
  * a **win-rate** (share of forward returns > 0) vs the unconditional base rate;
  * **one execution lag** — the month-t GGR print is released by the NGCB ~5 weeks later
    (during month t+1), so it is acted on at month t+1's close and forward returns start
    from t+1 (no look-ahead);
  * **one-way costs** x turnover for the "own the casino basket when GGR momentum is rising,
    else cash" timing overlay.

The decisive question is not whether casino stocks and Strip GGR co-move over the cycle (they
do — same demand) but whether a monthly GGR uptick **leads** the equities enough, and cleanly
enough, to be a tradable signal — rather than a backward-looking print the liquid, forward-
looking stocks have already discounted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)            # forward horizons in months


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def ggr_ttm(frame: pd.DataFrame) -> pd.Series:
    """Trailing-12-month sum of Strip GGR (a deseasonalised run-rate). Needs 12 months."""
    return frame["ggr"].rolling(12, min_periods=12).sum()


def ggr_momentum(frame: pd.DataFrame, k: int = 3) -> pd.Series:
    """k-month momentum of the trailing-12-month GGR run-rate: T_t / T_{t-k} - 1."""
    ttm = ggr_ttm(frame)
    return ttm / ttm.shift(k) - 1.0


def rising_mask(frame: pd.DataFrame, k: int = 3, thresh: float = 0.0) -> pd.Series:
    """Boolean: GGR momentum strictly above ``thresh`` (Strip run-rate RISING)."""
    return ggr_momentum(frame, k=k) > thresh


# --------------------------------------------------------------------------- #
# Forward returns (one-month execution lag)
# --------------------------------------------------------------------------- #
def forward_returns(frame: pd.DataFrame, months: int, lag: int = 1) -> pd.Series:
    """Forward ``months``-month basket return entered ``lag`` months after the signal.

    Signal known at month-end t (the GGR print is released during t+1) is acted on at
    month-end t+lag (no look-ahead); the return runs t+lag -> t+lag+months. NaN where the
    horizon overruns the tape.
    """
    b = frame["basket"]
    entry = b.shift(-lag)
    exit_ = b.shift(-lag - months)
    return exit_ / entry - 1.0


def split_returns(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
                  lag: int = 1):
    """(rising_fwd, falling_fwd, all_fwd) arrays of forward returns, NaNs dropped."""
    fwd = forward_returns(frame, months, lag=lag)
    rising = rising_mask(frame, k=k, thresh=thresh)
    ok = fwd.notna() & rising.notna()
    fwd, rising = fwd[ok], rising[ok].astype(bool)
    r = fwd[rising].values.astype(float)
    f = fwd[~rising].values.astype(float)
    a = fwd.values.astype(float)
    return r, f, a


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance). NaN if sample < 2."""
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
                   lag: int = 1, n_draws: int = 20_000, seed: int = 762) -> dict:
    """Small-sample placebo null for the bullish leading claim.

    Draw ``n_rising`` random months ``n_draws`` times and ask how often a random draw's mean
    forward return is **at least** the RISING set's mean (i.e. as *bullish* or more).
    p = P[random-draw mean >= rising mean]. A real leading signal => small p.
    """
    r, _f, a = split_returns(frame, months, k=k, thresh=thresh, lag=lag)
    kk = len(r)
    if kk == 0 or len(a) == 0:
        return {"k": 0, "rising_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    n = len(a)
    for i in range(n_draws):
        means[i] = a[rng.integers(0, n, size=kk)].mean()
    obs = float(r.mean())
    p = float((means >= obs).mean())
    return {"k": kk, "rising_mean": obs, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(frame: pd.DataFrame, months: int, k: int = 3, thresh: float = 0.0,
              lag: int = 1) -> dict:
    """Headline stats for one horizon: n, RISING vs FALLING vs base forward means and
    up-rates, the Welch t (RISING vs base), and the placebo p (leading signal)."""
    r, f, a = split_returns(frame, months, k=k, thresh=thresh, lag=lag)
    pl = placebo_pvalue(frame, months, k=k, thresh=thresh, lag=lag)
    return {
        "months": months,
        "n_rising": int(len(r)),
        "n_falling": int(len(f)),
        "rising_mean": float(r.mean()) if len(r) else float("nan"),
        "falling_mean": float(f.mean()) if len(f) else float("nan"),
        "base_mean": float(a.mean()) if len(a) else float("nan"),
        "rising_uprate": float((r > 0).mean()) if len(r) else float("nan"),
        "base_uprate": float((a > 0).mean()) if len(a) else float("nan"),
        "t": welch_t(r, a),
        "p_placebo": pl["p_value"],
    }


def lead_lag(frame: pd.DataFrame, k: int = 3, leads=range(-6, 7)) -> pd.Series:
    """Correlation of GGR momentum at t with the basket return over [t+L, t+L+1].

    L > 0 => GGR momentum *leads* the stocks (a genuine leading signal would peak at L > 0
    with a POSITIVE correlation); L < 0 => GGR momentum *lags* the stocks (a coincident or
    lagging echo — the stocks moved first). Returns a Series indexed by lead L (months)."""
    mom = ggr_momentum(frame, k=k)
    b = frame["basket"]
    out = {}
    for L in leads:
        fwd = b.shift(-L - 1) / b.shift(-L) - 1.0
        s = pd.concat([mom, fwd], axis=1).dropna()
        out[L] = (float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1])
                  if len(s) > 3 else float("nan"))
    return pd.Series(out, name="corr")


# --------------------------------------------------------------------------- #
# Tradability — "own the casino basket when GGR momentum is rising" overlay
# --------------------------------------------------------------------------- #
def timing_overlay(frame: pd.DataFrame, k: int = 3, thresh: float = 0.0, lag: int = 1,
                   cost_bps: float = 10.0) -> dict:
    """Own the casino basket when GGR momentum is RISING, sit in cash when FALLING.

    One-month execution lag; one-way cost ``cost_bps`` charged on each switch (turnover
    counted one-way x NAV). Compares the overlay's monthly return stream to buy-and-hold of
    the basket on the same months, reporting annualised mean, vol, Sharpe (excess-of-zero,
    since the cash leg earns 0 here — a conservative, clearly-labelled simplification), and
    the number of switches. Gross and net both reported.
    """
    b = frame["basket"]
    rising = rising_mask(frame, k=k, thresh=thresh)
    # signal known at t acts from t+lag: position for month t+lag = (rising at t)
    pos = rising.astype(float).shift(lag)
    df = pd.DataFrame({"r": b.pct_change(), "pos": pos}).dropna()
    bh = df["r"]
    switches = df["pos"].diff().abs().fillna(0.0)
    c = cost_bps / 1e4
    gross = df["pos"] * bh
    net = gross - switches * c
    n_sw = int((switches > 0).sum())

    def _ann(x):
        mu = x.mean() * 12.0
        vol = x.std(ddof=1) * np.sqrt(12.0)
        sharpe = mu / vol if vol > 0 else float("nan")
        return mu, vol, sharpe

    bh_mu, bh_vol, bh_sh = _ann(bh)
    g_mu, g_vol, g_sh = _ann(gross)
    n_mu, n_vol, n_sh = _ann(net)
    return {
        "n_months": int(len(df)), "n_switches": n_sw,
        "bh_mean": bh_mu, "bh_vol": bh_vol, "bh_sharpe": bh_sh,
        "overlay_gross_mean": g_mu, "overlay_gross_sharpe": g_sh,
        "overlay_net_mean": n_mu, "overlay_net_vol": n_vol, "overlay_net_sharpe": n_sh,
        "cost_bps": cost_bps,
    }
