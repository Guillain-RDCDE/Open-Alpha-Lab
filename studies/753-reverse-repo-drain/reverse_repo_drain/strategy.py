"""Strategy + inference for Study 753 — Reverse-Repo-Drain.

The claim, operationalised on a monthly frame of ``rrp`` (the ON RRP proxy, $B) and ``spy``
(month-end close):

    When the Fed's ON RRP facility is **draining** — money-market cash leaving the RRP and
    (the story goes) flowing into risk assets — it marks a **risk-on** regime, so be long
    stocks. A *filling* RRP drains liquidity out of markets, so be cautious.

We operationalise "draining" as the trailing ``k``-month change in the RRP level being
negative (the balance is lower than it was ``k`` months ago), known at the close of month
``t`` and acted on at ``t+1`` (a one-month execution lag — no look-ahead).

We test it three ways:

  * **Regime means.** Next-month SPY return in the *draining* regime vs the *filling*
    regime, and vs the unconditional monthly mean. The signal is the **spread** between
    regimes, not the raw draining mean (which mostly inherits the market's own up-drift).
  * **A block-bootstrap / placebo null.** RRP regimes are few and *long* (one big fill,
    one big drain), so an i.i.d. label shuffle would badly understate the null variance.
    We resample the regime-label sequence in contiguous blocks and ask how often a random
    block-labelling reproduces the observed spread — the honest small-sample test.
  * **A timing backtest, net of costs.** Hold SPY when the RRP is draining (long/flat, or
    long/short the filling months), one-month lag, one-way cost per switch, raced against
    buy-and-hold on a Sharpe basis (SPY total-return via yfinance auto-adjust; labelled).

The decisive caution is on the Signal axis: the RRP's entire meaningful history is a
**single fill-then-drain episode** (2021 ramp -> 2022 peak -> 2023-25 drain) that happens
to straddle one bear market and one bull market. Any "drain = risk-on" spread is an n=1
macro coincidence until a second episode exists to confirm it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_K = 3           # trailing months over which "draining" is measured
ANN = 12                # months per year


# --------------------------------------------------------------------------- #
# Returns & regime
# --------------------------------------------------------------------------- #
def monthly_returns(frame: pd.DataFrame) -> pd.Series:
    """Simple monthly SPY returns (month-over-month), aligned to the month-end index."""
    return frame["spy"].pct_change().dropna()


def draining(frame: pd.DataFrame, k: int = DEFAULT_K) -> pd.Series:
    """Boolean Series: was the ON RRP *draining* as of month ``t`` (level below ``t-k``)?

    Known at the close of month ``t`` (uses only the level then and ``k`` months earlier).
    The first ``k`` months carry no trailing change and are left **undefined** (NaN, so the
    caller drops them) rather than silently labelled "filling".
    """
    d = frame["rrp"].diff(k)
    return (d < 0).where(d.notna())


def regime_returns(frame: pd.DataFrame, k: int = DEFAULT_K,
                   lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Next-month SPY returns split by the *prior*-month drain regime (1-month exec lag).

    The drain flag at the close of month ``t`` is acted on in month ``t+lag`` (no
    look-ahead). Returns ``(drain, fill)``: forward monthly returns earned while the
    signalling RRP was draining / filling.
    """
    ret = frame["spy"].pct_change().shift(-1)         # return earned NEXT month
    sig = draining(frame, k=k)
    if lag > 1:
        sig = sig.shift(lag - 1)
    df = pd.DataFrame({"ret": ret, "drain": sig}).dropna()
    mask = df["drain"].astype(bool)
    drain = df.loc[mask, "ret"].values.astype(float)
    fill = df.loc[~mask, "ret"].values.astype(float)
    return drain, fill


def unconditional_returns(frame: pd.DataFrame) -> np.ndarray:
    """All monthly SPY returns (the base-rate distribution)."""
    return monthly_returns(frame).values.astype(float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of ``mean(sample) - mean(base)`` (unequal variance). NaN if either < 2."""
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def block_bootstrap_pvalue(frame: pd.DataFrame, k: int = DEFAULT_K, block: int = 6,
                           n_draws: int = 20_000, lag: int = 1, seed: int = 753) -> dict:
    """Block-placebo null for the regime spread (drain mean minus fill mean).

    RRP regimes are few and long, so an i.i.d. label shuffle would understate the null
    variance. We resample a regime-label sequence in contiguous blocks of ``block`` months
    (drawn to match the observed draining fraction), keep the return series fixed, and
    recompute the drain-minus-fill spread each draw. ``p = P[random spread >= observed]``.
    """
    ret = frame["spy"].pct_change().shift(-1)
    sig = draining(frame, k=k)
    if lag > 1:
        sig = sig.shift(lag - 1)
    df = pd.DataFrame({"ret": ret, "drain": sig}).dropna()
    r = df["ret"].values.astype(float)
    lab = df["drain"].astype(bool).values
    n = len(r)
    if n < 2 * block or lab.sum() == 0 or (~lab).sum() == 0:
        return {"obs_spread": float("nan"), "p_value": float("nan"), "n": n}
    obs = float(r[lab].mean() - r[~lab].mean())
    frac_drain = lab.mean()
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    spreads = np.empty(n_draws)
    for i in range(n_draws):
        blab = rng.random(n_blocks) < frac_drain
        lab_b = np.repeat(blab, block)[:n]
        if lab_b.all() or (~lab_b).all():
            spreads[i] = 0.0
            continue
        spreads[i] = r[lab_b].mean() - r[~lab_b].mean()
    p = float((spreads >= obs).mean())
    return {"obs_spread": obs, "placebo_spread": float(spreads.mean()),
            "p_value": p, "n": n, "k_drain": int(lab.sum())}


def summarize(frame: pd.DataFrame, k: int = DEFAULT_K, lag: int = 1) -> dict:
    """Headline regime stats: n in each regime, conditional means/win-rates, the
    unconditional base rate, the drain-minus-fill spread, Welch t, and the block-placebo p."""
    drain, fill = regime_returns(frame, k=k, lag=lag)
    base = unconditional_returns(frame)
    pl = block_bootstrap_pvalue(frame, k=k, lag=lag)
    return {
        "k": k,
        "n_drain": int(len(drain)),
        "n_fill": int(len(fill)),
        "drain_mean": float(drain.mean()) if len(drain) else float("nan"),
        "fill_mean": float(fill.mean()) if len(fill) else float("nan"),
        "drain_win": float((drain > 0).mean()) if len(drain) else float("nan"),
        "fill_win": float((fill > 0).mean()) if len(fill) else float("nan"),
        "base_mean": float(base.mean()),
        "base_win": float((base > 0).mean()),
        "spread": (float(drain.mean() - fill.mean())
                   if len(drain) and len(fill) else float("nan")),
        "t": welch_t(drain, fill),
        "p_placebo": pl["p_value"],
    }


# --------------------------------------------------------------------------- #
# Deployable timing strategy: own SPY only when the RRP is draining
# --------------------------------------------------------------------------- #
def timing_backtest(frame: pd.DataFrame, k: int = DEFAULT_K, cost_bps: float = 10.0,
                    lag: int = 1, allow_short: bool = False,
                    rf_annual: float = 0.0) -> dict:
    """Long/flat (or long/short) SPY-timing rule driven by the drain regime.

    Position for month ``m`` is decided by the RRP known ``lag`` months earlier: +1 when
    draining; 0 (long/flat, sit in cash at ``rf_annual``) or -1 (long/short) otherwise. A
    one-way ``cost_bps`` is charged on each change of position. Returns gross/net annualized
    return, vol and Sharpe for the rule and for buy-and-hold. SPY is yfinance auto-adjusted
    (**total-return**), labelled as such; ``rf_annual`` is the cash yield earned while flat
    (0 by default — a conservative excess-of-cash-vs-excess-of-cash comparison)."""
    ret = frame["spy"].pct_change()
    sig = draining(frame, k=k)                       # True / False / NaN (first k months)
    off = -1.0 if allow_short else 0.0
    pos = sig.map({True: 1.0, False: off}).shift(lag)  # NaN months drop out below
    rf_m = (1.0 + rf_annual) ** (1 / 12) - 1.0
    df = pd.DataFrame({"ret": ret, "pos": pos}).dropna()
    c = cost_bps / 1e4
    turn = df["pos"].diff().abs().fillna(df["pos"].abs())
    gross = df["pos"] * df["ret"] + (df["pos"] == 0).astype(float) * rf_m
    net = gross - turn * c
    bh = df["ret"]

    def _stats(s: pd.Series) -> dict:
        mu, sd = s.mean() * ANN, s.std(ddof=1) * np.sqrt(ANN)
        return {"ann_ret": float(mu), "ann_vol": float(sd),
                "sharpe": float(mu / sd) if sd > 0 else float("nan")}

    return {
        "n_months": int(len(df)),
        "n_turns": float(turn.sum()),
        "exposure": float((df["pos"] > 0).mean()),
        "gross": _stats(gross),
        "net": _stats(net),
        "buy_hold": _stats(bh),
        "cost_bps": cost_bps,
        "allow_short": allow_short,
    }
