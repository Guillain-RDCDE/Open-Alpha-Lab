"""Strategy + inference for Study 384 — ISM-PMI-Regime.

The folklore (on a PMI proxy here, since the true ISM survey is proprietary and FRED is
unreachable from this environment):

    Own equities **only when the manufacturing PMI is above 50** (expansion); sit in cash
    when it is below 50 (contraction). A reading above 50 is supposed to mark the regime in
    which stocks reliably pay you, and below 50 the regime to avoid.

We test it two ways:

  * **Conditional means.** Monthly SPY return in the PMI>50 regime vs the PMI<50 regime,
    vs the unconditional monthly mean. The signal is the *spread* between regimes, not the
    raw above-50 mean (which is mostly the market's own up-drift).
  * **A deployable timing strategy.** Hold SPY when last month's PMI was above 50, else
    hold cash; entry lagged one month (the PMI for month *t* is acted on in *t+1*, no
    look-ahead), one-way costs charged on each regime switch.

Inference:

  * a **Welch t** of the above-50 vs below-50 monthly returns (unequal variance);
  * a **block-bootstrap / placebo** null that respects monthly autocorrelation — shuffle
    the regime labels in blocks and ask how often a random labelling produces a spread as
    large as the real one (the honest small-sample test, since PMI regimes are few and long);
  * a **win-rate** in each regime vs the unconditional base rate;
  * **one-month execution lag** and **one-way costs** on the timing strategy.

The decisive question is whether the above-50 *excess* over the unconditional mean is
robust — or whether "own stocks above 50" is just "own stocks," dressed in a macro costume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLD = 50.0
TRADING_MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Returns & regime
# --------------------------------------------------------------------------- #
def monthly_returns(frame: pd.DataFrame) -> pd.Series:
    """Simple monthly SPY returns (month-over-month), aligned to the month-end index."""
    return frame["spy"].pct_change().dropna()


def regime_returns(frame: pd.DataFrame, threshold: float = THRESHOLD,
                   lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Next-month SPY returns split by *prior*-month PMI regime (1-month execution lag).

    The PMI reading at the close of month ``t`` is acted on in month ``t+lag`` (no
    look-ahead). Returns ``(above, below)``: the arrays of forward monthly returns earned
    while the signalling PMI was > / <= ``threshold``.
    """
    ret = frame["spy"].pct_change().shift(-1)              # return earned NEXT month
    pmi = frame["pmi"]
    # signal known at t, return earned at t+1: a `lag-1` extra shift gives the convention
    sig = (pmi > threshold)
    if lag > 1:
        sig = sig.shift(lag - 1)
    df = pd.DataFrame({"ret": ret, "above": sig}).dropna()
    above = df.loc[df["above"], "ret"].values.astype(float)
    below = df.loc[~df["above"], "ret"].values.astype(float)
    return above, below


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
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def block_bootstrap_pvalue(frame: pd.DataFrame, threshold: float = THRESHOLD,
                           block: int = 6, n_draws: int = 20_000, lag: int = 1,
                           seed: int = 384) -> dict:
    """Block-placebo null for the regime spread (above-50 mean minus below-50 mean).

    PMI regimes are few and long, so an i.i.d. label shuffle would understate the null
    variance. We instead resample the *regime label sequence* in contiguous blocks of
    ``block`` months (a circular block bootstrap of the labels), keep the return series
    fixed, and recompute the above-minus-below spread each draw. The p-value is the share
    of random block-labellings whose spread is at least the observed spread.
    """
    ret = frame["spy"].pct_change().shift(-1)
    pmi = frame["pmi"]
    sig = (pmi > threshold)
    if lag > 1:
        sig = sig.shift(lag - 1)
    df = pd.DataFrame({"ret": ret, "above": sig}).dropna()
    r = df["ret"].values.astype(float)
    lab = df["above"].values.astype(bool)
    n = len(r)
    if n < 2 * block or lab.sum() == 0 or (~lab).sum() == 0:
        return {"obs_spread": float("nan"), "p_value": float("nan"), "n": n}
    obs = float(r[lab].mean() - r[~lab].mean())
    k_above = int(lab.sum())
    frac_above = k_above / n
    rng = np.random.default_rng(seed)
    spreads = np.empty(n_draws)
    n_blocks = int(np.ceil(n / block))
    for i in range(n_draws):
        # build a random label sequence in blocks, preserving the above-fraction in expectation
        starts = rng.integers(0, n, size=n_blocks)
        seq = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        # random block labels drawn to match the above-fraction
        blab = rng.random(n_blocks) < frac_above
        lab_b = np.repeat(blab, block)[:n]
        m_a = r[lab_b].mean() if lab_b.any() else 0.0
        m_b = r[~lab_b].mean() if (~lab_b).any() else 0.0
        spreads[i] = m_a - m_b
    p = float((spreads >= obs).mean())
    return {"obs_spread": obs, "placebo_spread": float(spreads.mean()),
            "p_value": p, "n": n, "k_above": k_above}


def summarize(frame: pd.DataFrame, threshold: float = THRESHOLD, lag: int = 1) -> dict:
    """Headline regime stats: n in each regime, conditional means/win-rates, the
    unconditional base rate, the above-minus-below spread, Welch t, and the block-placebo p."""
    above, below = regime_returns(frame, threshold=threshold, lag=lag)
    base = unconditional_returns(frame)
    pl = block_bootstrap_pvalue(frame, threshold=threshold, lag=lag)
    return {
        "threshold": threshold,
        "n_above": int(len(above)),
        "n_below": int(len(below)),
        "above_mean": float(above.mean()) if len(above) else float("nan"),
        "below_mean": float(below.mean()) if len(below) else float("nan"),
        "above_win": float((above > 0).mean()) if len(above) else float("nan"),
        "below_win": float((below > 0).mean()) if len(below) else float("nan"),
        "base_mean": float(base.mean()),
        "base_win": float((base > 0).mean()),
        "spread": (float(above.mean() - below.mean())
                   if len(above) and len(below) else float("nan")),
        "t": welch_t(above, below),
        "p_placebo": pl["p_value"],
    }


# --------------------------------------------------------------------------- #
# Deployable timing strategy: own SPY only when prior-month PMI > 50
# --------------------------------------------------------------------------- #
def timing_strategy(frame: pd.DataFrame, threshold: float = THRESHOLD,
                    cost_bps: float = 10.0, lag: int = 1,
                    rf_annual: float = 0.0) -> dict:
    """Own SPY when last month's PMI > threshold, else hold cash; one-month lag, one-way
    costs on each switch.

    Returns gross & net annualised return, vol, Sharpe of the timing rule and of plain
    buy-and-hold, plus the number of regime switches. Costs are one-way (charged once per
    leg) on a NAV that is fully in or fully out: each switch is one round of turnover.
    ``rf_annual`` is the cash yield earned while out of the market (0 by default — a
    conservative, excess-of-cash-vs-excess-of-cash comparison uses rf=0 on both legs).
    """
    ret = frame["spy"].pct_change()
    pmi = frame["pmi"]
    # signal known at close of t-1 governs the return of t: shift the regime forward 1 month
    pos = (pmi > threshold).shift(lag).fillna(False).astype(float)
    rf_m = (1.0 + rf_annual) ** (1 / 12) - 1.0
    df = pd.DataFrame({"ret": ret, "pos": pos}).dropna()
    switches = int((df["pos"].diff().abs() > 0).sum())
    c = cost_bps / 1e4
    cost_series = df["pos"].diff().abs().fillna(0.0) * c   # pay c on each entry/exit leg
    strat_gross = df["pos"] * df["ret"] + (1 - df["pos"]) * rf_m
    strat_net = strat_gross - cost_series
    bh = df["ret"]

    def ann(series):
        mu = series.mean() * 12
        vol = series.std(ddof=1) * np.sqrt(12)
        sharpe = (series.mean() / series.std(ddof=1) * np.sqrt(12)
                  if series.std(ddof=1) > 0 else float("nan"))
        return float(mu), float(vol), float(sharpe)

    g_mu, g_vol, g_sh = ann(strat_gross)
    n_mu, n_vol, n_sh = ann(strat_net)
    b_mu, b_vol, b_sh = ann(bh)
    return {
        "n_months": int(len(df)),
        "switches": switches,
        "gross_ann": g_mu, "gross_vol": g_vol, "gross_sharpe": g_sh,
        "net_ann": n_mu, "net_vol": n_vol, "net_sharpe": n_sh,
        "bh_ann": b_mu, "bh_vol": b_vol, "bh_sharpe": b_sh,
        "cost_bps": cost_bps,
    }
