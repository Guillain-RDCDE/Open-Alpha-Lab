"""Strategy + inference for Study 761 — Hotel-RevPAR.

The believers' rule (travel-cycle timing): **hotel RevPAR is a leading gauge of the
lodging cycle — when RevPAR momentum turns up, be long hotel REITs, because accelerating
travel demand isn't fully priced yet.** Operationalised on a monthly frame of ``revpar``
(the cited RevPAR proxy, dollars) and ``px`` (a month-end hotel-equity price — HST or the
lodging-REIT basket, total-return):

    Let ``m_t = log(revpar_t / revpar_{t-12})`` be RevPAR's **YoY log momentum** (YoY
    differencing kills the strong hotel seasonality). An "UPCYCLE" month is ``m_t > thr``
    (RevPAR growing year-on-year). The believers say the forward hotel-equity return is
    then ELEVATED relative to the unconditional mean.

We test it four ways:

  * **Conditional vs unconditional forward returns** — mean H-month forward hotel return
    in UPCYCLE vs DOWNCYCLE vs the unconditional base rate, with a **Welch two-sample t**
    and a **placebo / randomization null** sized to the event count.
  * **A predictive regression with HAC (Newey-West) inference** — regress the forward
    H-month return on ``m_t``; the slope's **HAC t** is the direct "does momentum *lead*?"
    test and the desk's |t| >= 2 bar. Overlapping windows ⇒ Newey-West is mandatory.
  * **A lead-lag cross-correlation** — corr of RevPAR momentum at ``t`` with the hotel
    return over ``[t+L, t+L+1]``. A genuine leading indicator peaks at **L > 0**; if it
    peaks at **L <= 0** the stock is *coincident-or-leading* the gauge, not the reverse.
  * **A timing backtest, net of costs** — long the hotel tape when RevPAR momentum > 0,
    else flat (or short), one execution lag, one-way cost per turn, raced against
    buy-and-hold on a Sharpe basis.

Release lag (documented): STR publishes reference-month-``t`` RevPAR ~20 days into month
``t+1``, so it is public by the close of month ``t+1``. We enter at that close — a
**one-month execution lag on the reference month** — and the forward return runs from
there, strictly after the print. Conservative and stated.

The decisive question is not whether hotel stocks and RevPAR move *together* over the
cycle (they must — same demand) but whether the RevPAR print **leads** the tape enough,
and cleanly enough, to be a tradable signal rather than a coincident-or-lagging echo of
what the equity already discounted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)            # forward horizons in months
ANN = 12


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def revpar_momentum(frame: pd.DataFrame, k: int = 12) -> pd.Series:
    """RevPAR YoY log momentum: log(revpar_t / revpar_{t-k}) (k=12 kills seasonality)."""
    r = frame["revpar"].astype(float)
    return np.log(r) - np.log(r.shift(k))


def upcycle_mask(frame: pd.DataFrame, k: int = 12, thr: float = 0.0) -> pd.Series:
    """Boolean: RevPAR YoY momentum strictly above ``thr`` (travel UPCYCLE)."""
    return revpar_momentum(frame, k=k) > thr


# --------------------------------------------------------------------------- #
# Forward returns (one-month execution lag on the reference month)
# --------------------------------------------------------------------------- #
def forward_returns(frame: pd.DataFrame, months: int, lag: int = 1) -> pd.Series:
    """Forward ``months``-month hotel return entered ``lag`` months after reference month.

    RevPAR for reference month ``t`` is public by the close of month ``t+lag`` (lag=1:
    STR's ~mid-t+1 release); we enter at that close and the return runs to
    ``t+lag+months`` — strictly after the print (no look-ahead). NaN where the horizon
    overruns the tape.
    """
    px = frame["px"]
    entry = px.shift(-lag)
    exit_ = px.shift(-lag - months)
    return exit_ / entry - 1.0


def split_returns(frame: pd.DataFrame, months: int, k: int = 12, thr: float = 0.0,
                  lag: int = 1):
    """(upcycle_fwd, downcycle_fwd, all_fwd) arrays of forward returns, NaNs dropped."""
    fwd = forward_returns(frame, months, lag=lag)
    up = upcycle_mask(frame, k=k, thr=thr)
    ok = fwd.notna() & up.notna()
    fwd, up = fwd[ok], up[ok].astype(bool)
    return (fwd[up].values.astype(float), fwd[~up].values.astype(float),
            fwd.values.astype(float))


# --------------------------------------------------------------------------- #
# Inference — Welch t, placebo null, Newey-West HAC regression
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


def newey_west_t(x: np.ndarray, y: np.ndarray, lags: int) -> dict:
    """OLS y = a + b*x with Newey-West (HAC) SE on the slope; returns b, t, n, r2.

    Overlapping forward-return windows induce strong serial correlation, so the slope's
    ordinary SE is far too small; HAC (Bartlett kernel, ``lags``) is the honest inference.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < lags + 3:
        return {"beta": float("nan"), "t": float("nan"), "n": n, "r2": float("nan")}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    # Newey-West meat: S = sum_{l=-L..L} w_l * (X_t' u_t)(X_{t-l}' u_{t-l})
    u = resid
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_b = np.sqrt(cov[1, 1])
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / tss if tss > 0 else float("nan")
    return {"beta": float(beta[1]), "t": float(beta[1] / se_b) if se_b > 0 else float("nan"),
            "n": n, "r2": float(r2)}


def predictive_regression(frame: pd.DataFrame, months: int, k: int = 12,
                          lag: int = 1) -> dict:
    """HAC-t of forward H-month hotel return regressed on RevPAR YoY momentum.

    The direct leading-indicator test: does a higher RevPAR momentum *predict* a higher
    forward return? HAC lag = ``months`` (the overlap length). |t| >= 2 with beta > 0
    would support the claim.
    """
    mom = revpar_momentum(frame, k=k)
    fwd = forward_returns(frame, months, lag=lag)
    df = pd.concat([mom.rename("m"), fwd.rename("f")], axis=1).dropna()
    return newey_west_t(df["m"].values, df["f"].values, lags=max(months, 1))


def placebo_pvalue(frame: pd.DataFrame, months: int, k: int = 12, thr: float = 0.0,
                   lag: int = 1, n_draws: int = 20_000, seed: int = 761) -> dict:
    """Small-sample placebo null for the "UPCYCLE months pay more" claim.

    Draw ``n_up`` random months ``n_draws`` times; p = P[random-draw mean >= upcycle
    mean]. A real bullish signal => small p."""
    up, _dn, a = split_returns(frame, months, k=k, thr=thr, lag=lag)
    kk = len(up)
    if kk == 0 or len(a) == 0:
        return {"k": 0, "up_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    n = len(a)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = a[rng.integers(0, n, size=kk)].mean()
    obs = float(up.mean())
    return {"k": kk, "up_mean": obs, "placebo_mean": float(means.mean()),
            "p_value": float((means >= obs).mean())}


def summarize(frame: pd.DataFrame, months: int, k: int = 12, thr: float = 0.0,
              lag: int = 1) -> dict:
    """Headline stats for one horizon: n, UPCYCLE/DOWNCYCLE/base forward means & win-rates,
    the Welch t (UPCYCLE vs base), the HAC regression t, and the placebo p."""
    up, dn, a = split_returns(frame, months, k=k, thr=thr, lag=lag)
    reg = predictive_regression(frame, months, k=k, lag=lag)
    pl = placebo_pvalue(frame, months, k=k, thr=thr, lag=lag)
    return {
        "months": months,
        "n_up": int(len(up)),
        "n_dn": int(len(dn)),
        "up_mean": float(up.mean()) if len(up) else float("nan"),
        "dn_mean": float(dn.mean()) if len(dn) else float("nan"),
        "base_mean": float(a.mean()) if len(a) else float("nan"),
        "up_win": float((up > 0).mean()) if len(up) else float("nan"),
        "base_win": float((a > 0).mean()) if len(a) else float("nan"),
        "t_welch": welch_t(up, a),
        "beta_hac": reg["beta"],
        "t_hac": reg["t"],
        "p_placebo": pl["p_value"],
    }


def lead_lag(frame: pd.DataFrame, k: int = 12, leads=range(-6, 7)) -> pd.Series:
    """Corr of RevPAR momentum at t with the hotel return over [t+L, t+L+1].

    L < 0 => momentum *lags* the equity (the stock led — coincident/lagging gauge);
    L > 0 => momentum *leads* the equity (a genuine early signal). A leading indicator
    peaks at L > 0. Returns a Series indexed by lead L (months).
    """
    mom = revpar_momentum(frame, k=k)
    px = frame["px"]
    out = {}
    for L in leads:
        fwd = px.shift(-L - 1) / px.shift(-L) - 1.0
        s = pd.concat([mom, fwd], axis=1).dropna()
        out[L] = (float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1])
                  if len(s) > 3 else float("nan"))
    return pd.Series(out, name="corr")


# --------------------------------------------------------------------------- #
# Tradability — the "long hotels when RevPAR momentum is up" timing overlay
# --------------------------------------------------------------------------- #
def timing_backtest(frame: pd.DataFrame, k: int = 12, thr: float = 0.0, lag: int = 1,
                    cost_bps: float = 10.0, allow_short: bool = False) -> dict:
    """Long hotels when RevPAR momentum > thr; flat (or short) otherwise.

    One-month execution lag on the reference month; one-way cost ``cost_bps`` charged on
    each position change (turnover one-way × NAV). Sharpe is excess-of-zero (the flat
    leg earns 0 here — a conservative, clearly-labelled simplification that *flatters* the
    timing rule). Gross and net both reported; the hotel tape is total-return (labelled).
    """
    px = frame["px"]
    ret = px.pct_change()
    up = upcycle_mask(frame, k=k, thr=thr)
    pos_raw = np.where(up, 1.0, (-1.0 if allow_short else 0.0))
    # signal for reference month t acted from t+lag
    pos = pd.Series(pos_raw, index=frame.index).shift(lag)
    df = pd.DataFrame({"r": ret, "pos": pos}).dropna()
    turn = df["pos"].diff().abs().fillna(df["pos"].abs())
    c = cost_bps / 1e4
    gross = df["pos"] * df["r"]
    net = gross - turn * c

    def _ann(x):
        mu = x.mean() * ANN
        vol = x.std(ddof=1) * np.sqrt(ANN)
        return {"ann_ret": float(mu), "ann_vol": float(vol),
                "sharpe": float(mu / vol) if vol > 0 else float("nan")}

    return {
        "n_months": int(len(df)),
        "n_turns": float(turn.sum()),
        "exposure": float((df["pos"] != 0).mean()),
        "gross": _ann(gross),
        "net": _ann(net),
        "buy_hold": _ann(df["r"]),
        "cost_bps": cost_bps,
        "allow_short": allow_short,
    }
