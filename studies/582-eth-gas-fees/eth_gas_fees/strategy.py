"""The engine and its honest controls — Study 582 (ETH-Gas-Fees).

The claim, steelmanned: Ethereum **gas fees** are the market-clearing price of blockspace. When
they spike, the chain is congested because activity has surged — mints, DEX swaps, leverage,
memecoin manias. That euphoric congestion is read as a **contrarian top signal**: pay-anything
behaviour marks the blow-off, so *forward* ETH returns should be LOW after a gas spike. The
bullish counter-read: high gas is genuine demand for blockspace and is neutral-to-positive for
forward returns. This module measures which (if either) is true, on a synthetic tape (real daily
gas data is out of reach for a no-key retail stack — see ``data.py``).

The apparatus, the desk's standard kit for a timing/euphoria-proxy claim:

1. **The gas spike.** A backward-looking standardised deviation of daily gas from its trailing
   30-day median (MAD-scaled), so it is public at day *t*. Higher = gas spiking vs its baseline.

2. **The contrarian overlay.** Short ETH (-1) when the gas spike sits in its top decile
   ("euphoric congestion"), flat otherwise — the tradable distillation of the top-signal claim.
   Its HAC *t* decides the (synthetic) signal reading. We also report the *long-only* mirror
   (buy the spike, the bullish read) so the sign is unambiguous.

3. **The forward-return slope.** OLS of the forward ETH return on the gas spike; a *negative*
   slope is the contrarian top signal, a *positive* slope is the genuine-demand read.

4. **A shuffled-gas placebo null.** Permute the gas spike against forward returns; a real signal
   sits in the tail, noise does not.

5. **Costs + borrow.** The overlay pays one-way turnover on every position change; short days pay
   a crypto borrow/funding fee (perp funding + spot borrow run into the high hundreds of bps/yr).

6. **A seed-robust synthetic positive control (>= 20 seeds).** Plant the top signal
   (``top_signal_beta < 0``) and confirm the slope-*t* goes negative and past the bar, and stays
   flat at the null — proving the engine catches a planted effect and does not manufacture one.

Execution convention: the gas spike at day *t* is trailing-only (public at day-*t* close); the
outcome is the forward return over (t, t+H]. ``forward_returns`` shifts the realised return so no
future data enters the signal — one documented execution lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365
SPIKE_WINDOW = 30
TOP_DECILE = 0.90


# ---------------------------------------------------------------------------
# The gas spike (trailing, public at day t)
# ---------------------------------------------------------------------------
def gas_spike(gas: pd.Series, window: int = SPIKE_WINDOW) -> pd.Series:
    """Standardised trailing gas spike: (gas - rolling median) / (1.4826 * rolling MAD).

    Backward-looking only, so it is public at the close of day *t*. Higher = gas is spiking
    relative to its recent baseline.
    """
    med = gas.rolling(window, min_periods=window // 2).median()
    mad = (gas - med).abs().rolling(window, min_periods=window // 2).median()
    scale = (mad * 1.4826).replace(0.0, np.nan)
    return ((gas - med) / scale).rename("gas_spike")


# ---------------------------------------------------------------------------
# Forward returns (the execution lag)
# ---------------------------------------------------------------------------
def build_frame(df: pd.DataFrame, horizon: int = 1, window: int = SPIKE_WINDOW) -> pd.DataFrame:
    """Attach the trailing gas spike and the H-day FORWARD ETH return to each row.

    ``df`` has columns ``eth_price`` and ``gas_gwei`` on a daily index. The spike at row *t* uses
    only data up to *t*; the forward return is price[t+H]/price[t] - 1, shifted back so it sits on
    row *t* (the signal never sees its own outcome). Rows without a full trailing window or a
    forward return are dropped. Returns columns ``gas_spike`` and ``fwd_ret``.
    """
    out = df.copy()
    out["gas_spike"] = gas_spike(out["gas_gwei"], window)
    fwd = out["eth_price"].shift(-horizon) / out["eth_price"] - 1.0
    out["fwd_ret"] = fwd
    return out[["gas_spike", "fwd_ret"]].dropna()


# ---------------------------------------------------------------------------
# The contrarian top-signal overlay (short on euphoric congestion)
# ---------------------------------------------------------------------------
def spike_threshold(fr: pd.DataFrame, q: float = TOP_DECILE) -> float:
    """The quantile cutoff of the gas spike that defines 'euphoric congestion'."""
    return float(fr["gas_spike"].quantile(q))


def contrarian_position(fr: pd.DataFrame, q: float = TOP_DECILE) -> pd.Series:
    """Target position: short ETH (-1) when gas spike is in its top decile, flat otherwise."""
    thr = spike_threshold(fr, q)
    pos = pd.Series(0.0, index=fr.index, name="pos")
    pos[fr["gas_spike"] >= thr] = -1.0
    return pos


def overlay_returns(fr: pd.DataFrame, q: float = TOP_DECILE) -> pd.Series:
    """Gross daily return of the contrarian top-signal overlay: position(t) * fwd_ret(t)."""
    return (contrarian_position(fr, q) * fr["fwd_ret"]).rename("overlay")


def spike_vs_calm(fr: pd.DataFrame, q: float = TOP_DECILE) -> dict:
    """Mean forward return on spike days vs calm days, and the two-sample (Welch) t.

    The contrarian claim predicts spike-day forward returns < calm-day forward returns (a negative
    spike-minus-calm gap). Returns both means, the gap, its Welch t, and the counts.
    """
    thr = spike_threshold(fr, q)
    spike = fr["fwd_ret"][fr["gas_spike"] >= thr].to_numpy(dtype=float)
    calm = fr["fwd_ret"][fr["gas_spike"] < thr].to_numpy(dtype=float)
    na, nb = len(spike), len(calm)
    if na < 2 or nb < 2:
        return {"spike_mean": float("nan"), "calm_mean": float("nan"),
                "gap": float("nan"), "t": float("nan"), "n_spike": na, "n_calm": nb}
    va, vb = spike.var(ddof=1), calm.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    gap = spike.mean() - calm.mean()
    return {
        "spike_mean": float(spike.mean()),
        "calm_mean": float(calm.mean()),
        "gap": float(gap),
        "t": float(gap / se) if se > 0 else float("nan"),
        "n_spike": na,
        "n_calm": nb,
    }


# ---------------------------------------------------------------------------
# The forward-return slope on the gas spike (the sign IS the claim)
# ---------------------------------------------------------------------------
def spike_slope(fr: pd.DataFrame) -> dict:
    """OLS of forward ETH return on the gas spike, with a Newey-West HAC t.

    Slope < 0 = the contrarian top signal (gas spikes precede low forward returns). Slope > 0 =
    the genuine-demand read. Returns the slope (per unit gas spike), its HAC t, and the n.
    """
    x = fr["gas_spike"].to_numpy(dtype=float)
    y = fr["fwd_ret"].to_numpy(dtype=float)
    n = len(y)
    if n < 6 or x.std() == 0:
        return {"slope": float("nan"), "tstat": float("nan"), "n": n}
    xc = x - x.mean()
    denom = float(xc @ xc)
    beta = float((xc @ (y - y.mean())) / denom)
    resid = (y - y.mean()) - beta * xc
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    xe = xc * resid
    s = float(xe @ xe)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(xe[k:] @ xe[:-k])
    var_beta = s / (denom ** 2)
    tstat = beta / np.sqrt(var_beta) if var_beta > 0 else float("nan")
    return {"slope": float(beta), "tstat": float(tstat), "n": int(n)}


# ---------------------------------------------------------------------------
# Shuffled-gas placebo null
# ---------------------------------------------------------------------------
def placebo_pvalue(fr: pd.DataFrame, n_perm: int = 2000, seed: int = 582,
                   q: float = TOP_DECILE) -> float:
    """Label-shuffle placebo p-value for the spike-minus-calm gap.

    Permute the gas-spike labels against forward returns; the p-value is the fraction of shuffles
    whose |gap| >= the observed |gap|. A real timing signal sits in the tail.
    """
    if len(fr) < 20:
        return float("nan")
    obs = abs(spike_vs_calm(fr, q)["gap"])
    rng = np.random.default_rng(seed)
    spikevals = fr["gas_spike"].to_numpy(dtype=float)
    y = fr["fwd_ret"].to_numpy(dtype=float)
    thr = np.quantile(spikevals, q)
    n = len(y)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        sp = spikevals[perm]
        mask = sp >= thr
        if mask.sum() < 2 or (~mask).sum() < 2:
            continue
        gap = y[mask].mean() - y[~mask].mean()
        if abs(gap) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow on the short overlay
# ---------------------------------------------------------------------------
def net_overlay_returns(fr: pd.DataFrame, one_way_bps: float = 10.0,
                        borrow_bps_ann: float = 1000.0, q: float = TOP_DECILE) -> pd.Series:
    """Contrarian overlay return net of one-way turnover and short borrow/funding.

    On every change in target position we pay ``one_way_bps`` x |delta position| x NAV (one-way x
    NAV). On any short day we additionally pay ``borrow_bps_ann / 365`` of NAV — crypto shorts are
    expensive (perp funding + spot borrow), so the default 1000 bps/yr is realistic, not punitive.
    """
    pos = contrarian_position(fr, q)
    gross = pos * fr["fwd_ret"]
    dpos = pos.diff().abs().fillna(pos.abs())
    trade_cost = dpos * one_way_bps * 1e-4
    borrow_cost = (pos < 0).astype(float) * (borrow_bps_ann * 1e-4) / PERIODS_PER_YEAR
    return (gross - trade_cost - borrow_cost).rename("net_overlay")


# ---------------------------------------------------------------------------
# Summary (Newey-West HAC t-stat) — matches the desk's timing-study kit
# ---------------------------------------------------------------------------
def summarize(series: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    """Headline statistics for a daily return series (HAC t-stat)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {"mean": float("nan"), "vol": float("nan"), "sharpe": float("nan"),
                "tstat": float("nan"), "hit_rate": float("nan"), "n": n}
    mu = float(r.mean())
    std = float(r.std(ddof=1))
    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean": mu,
        "vol": std,
        "sharpe": (mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Robustness sweep — multiple forward horizons
# ---------------------------------------------------------------------------
def horizon_sweep(df: pd.DataFrame, horizons=(1, 3, 7, 14, 30), q: float = TOP_DECILE) -> pd.DataFrame:
    """Re-run the spike-minus-calm gap + slope-t over several forward horizons.

    A real top signal should hold sign across horizons; a fluke will not. Returns a frame indexed
    by horizon with the gap, its Welch t, and the HAC slope-t.
    """
    rows = []
    for h in horizons:
        fr = build_frame(df, horizon=h)
        sc = spike_vs_calm(fr, q)
        sl = spike_slope(fr)
        rows.append({"horizon": h, "gap": sc["gap"], "gap_t": sc["t"], "slope_t": sl["tstat"]})
    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# Seed-robust synthetic positive control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, top_signal_beta: float, n_seeds: int = 25,
                     base_seed: int = 582, horizon: int = 1) -> float:
    """Average the forward-return slope-t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean slope-t across seeds for a
    planted ``top_signal_beta`` (negative = the contrarian top signal).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        df, _ = data_mod.synthetic_series(top_signal_beta=top_signal_beta, seed=s)
        fr = build_frame(df, horizon=horizon)
        ts.append(spike_slope(fr)["tstat"])
    return float(np.nanmean(ts))
