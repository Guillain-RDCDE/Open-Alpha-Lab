"""The Fibonacci-retracement engine + its honest controls — Study 541.

The chartist claim, stated so it can be falsified: *after a clean price swing, when the pullback
retraces to a **Fibonacci** fraction of that swing (0.236 / 0.382 / 0.5 / 0.618), price tends to
reverse there* — the pullback stops at the Fib level and the original trend resumes. Practitioners
treat 38.2% and 61.8% ("the golden ratio") as the levels that "matter".

We test that mechanically and honestly:

1. **ZigZag swing detector** — the standard tool for marking the prior swing. A new pivot is
   confirmed only after price reverses by ``pct`` from the last extreme; the pivot is *known* at
   the confirmation bar (no look-ahead).

2. **The realised retracement of each swing.** From the alternating pivot sequence P0, P1, P2, ...
   each three-pivot window is a swing P0->P1 and a pullback P1->P2. The pullback bottom is the
   *confirmed* pivot P2, so the realised retracement fraction ``retr = (P1-P2)/(P1-P0)`` is a real,
   look-ahead-free number known at P2's confirmation bar. No line-scanning, no double counting:
   each pullback contributes exactly one retracement. The chartist bet at P2 is **reversal** — the
   trend resumes in the P0->P1 direction; we enter **one bar after** P2 is confirmed (one-day
   execution lag) and hold ``horizon`` days.

3. **The Fibonacci arm vs the placebo arm.** Bin the swings by their realised retracement: the
   **Fib arm** keeps swings that landed near 0.236/0.382/0.5/0.618; the **placebo arm** keeps
   swings that landed near arbitrary non-Fibonacci fractions *interleaved in the same depth span*
   (0.31/0.44/0.56). Same machinery, same depth range — only the exact fractions differ. If
   Fibonacci levels are special, the Fib arm's reversal edge must *beat* the placebo arm's; if any
   round-ish level is as good, they match — the whole point of the study.

4. **Inference.** For each arm: a one-sample *t* (and a HAC / Newey-West *t*) of the
   direction-signed forward return vs 0; a win-rate vs the 50% base rate; and a same-bars
   random-direction **coin placebo** ("does betting reversal beat a coin at the same swings?"). The
   headline comparison is the **Fib-minus-placebo edge** with a Welch two-sample *t*.

5. **Costs.** One-way cost per leg × turnover (a round trip pays it twice).

6. **A synthetic positive control.** A deterministic world where reversal-at-Fib is *planted*
   (``fib_pull > 0``); the engine must recover a positive Fib-arm *t* and a positive Fib-minus-
   placebo edge, and stay flat at the null — averaged over >= 20 seeds (house rule).

No look-ahead: a pivot is confirmed at a bar strictly after its extreme; the retracement of a swing
is only known once P2 is confirmed; and we enter one bar after that. Every input to a trade is
public before the trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)

FIB_LEVELS = (0.236, 0.382, 0.5, 0.618)
PLACEBO_LEVELS = (0.31, 0.44, 0.56)


# --------------------------------------------------------------------------- #
# ZigZag swing detector (marks the prior swing)
# --------------------------------------------------------------------------- #
def zigzag(close: np.ndarray, pct: float = 0.05) -> pd.DataFrame:
    """Percentage ZigZag pivots on a close series (look-ahead-free confirmation bar).

    Walk forward tracking the running extreme since the last confirmed pivot. A new pivot is
    *confirmed* the first time price reverses by at least ``pct`` from that extreme; the pivot is
    stamped at the bar of the extreme (``piv_idx``) but only *known* at the reversal bar
    (``conf_idx``).

    Returns a DataFrame of alternating pivots: ``piv_idx`` (bar of the extreme), ``conf_idx``
    (bar it was confirmed = known), ``price`` (extreme price), ``kind`` (+1 high, -1 low).
    """
    n = len(close)
    if n < 3:
        return pd.DataFrame(columns=["piv_idx", "conf_idx", "price", "kind"])

    pivots = []
    hi_idx = lo_idx = 0
    hi_px = lo_px = close[0]
    direction = 0  # 0 = undetermined, +1 = uptrend, -1 = downtrend

    for i in range(1, n):
        px = close[i]
        if direction >= 0 and px > hi_px:
            hi_px, hi_idx = px, i
        if direction <= 0 and px < lo_px:
            lo_px, lo_idx = px, i

        if direction == 0:
            if px <= hi_px * (1.0 - pct):
                pivots.append((hi_idx, i, hi_px, +1))
                direction = -1
                lo_px, lo_idx = px, i
            elif px >= lo_px * (1.0 + pct):
                pivots.append((lo_idx, i, lo_px, -1))
                direction = +1
                hi_px, hi_idx = px, i
        elif direction > 0:
            if px <= hi_px * (1.0 - pct):
                pivots.append((hi_idx, i, hi_px, +1))
                direction = -1
                lo_px, lo_idx = px, i
        else:
            if px >= lo_px * (1.0 + pct):
                pivots.append((lo_idx, i, lo_px, -1))
                direction = +1
                hi_px, hi_idx = px, i

    piv = pd.DataFrame(pivots, columns=["piv_idx", "conf_idx", "price", "kind"])
    return piv.drop_duplicates(subset=["piv_idx"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Retracement touches (the tradable event)
# --------------------------------------------------------------------------- #
def swing_retracements(pivots: pd.DataFrame) -> pd.DataFrame:
    """One row per completed swing+pullback: how deep the pullback retraced, and where price is.

    A ZigZag makes an alternating pivot sequence P0, P1, P2, P3, ... Take a window of three
    consecutive pivots P0 -> P1 -> P2: P0->P1 is the *swing* (an impulse leg) and P1->P2 is the
    *pullback* against it. The pullback bottom is the confirmed pivot **P2** — so the realised
    retracement fraction of the swing is

        retr = (P1_price - P2_price) / (P1_price - P0_price)   (>0 for a genuine pullback)

    is fully known at **P2's confirmation bar** — a real, look-ahead-free number (no scanning, no
    line-crossing double counting; each pullback contributes exactly one retracement). At that bar
    the chartist bets **reversal**: the trend resumes in the P0->P1 direction. We record the resume
    direction and the signal bar (P2 confirmation). The caller bins these by whether ``retr`` sits
    near a Fibonacci level or a placebo level and compares the forward returns of the two bins.

    Returns columns: ``sig_idx`` (P2 confirmation bar = known), ``dir`` (+1/-1 resume direction),
    ``retr`` (realised retracement fraction).
    """
    out = []
    for k in range(2, len(pivots)):
        p0, p1, p2 = pivots.iloc[k - 2], pivots.iloc[k - 1], pivots.iloc[k]
        span = p1["price"] - p0["price"]
        if span == 0:
            continue
        # Only treat P0->P1 as a genuine *impulse* swing when it is at least as large as the leg
        # before it (an expansion, not itself a retracement of an earlier move). This is the
        # standard "measure the retracement of the impulse leg" convention and it keeps each
        # pullback attributed to exactly one swing (no double counting a leg as both swing and
        # pullback of the adjacent triple).
        if k >= 3:
            prev_leg = abs(p0["price"] - pivots.iloc[k - 3]["price"])
            if abs(span) < prev_leg:
                continue
        direction = 1 if span > 0 else -1
        pull = p1["price"] - p2["price"]
        # a genuine retracement is opposite the swing and *shallower* than it: 0 < retr < 1. Deeper
        # "retracements" (>= 1) are a full reversal, not a pullback of the swing, and are dropped —
        # the standard chartist definition and the honest one (no >100% retracements).
        retr = pull / span
        if not (0.0 < retr < 1.0):
            continue
        out.append({"sig_idx": int(p2["conf_idx"]), "dir": int(direction), "retr": float(retr)})
    return pd.DataFrame(out, columns=["sig_idx", "dir", "retr"])


def _near_any(retr: float, levels, tol: float) -> bool:
    return any(abs(retr - lv) <= tol for lv in levels)


def bin_touches(retr_df: pd.DataFrame, levels, tol: float = 0.025) -> pd.DataFrame:
    """Keep the swings whose realised retracement landed within ``tol`` of any of ``levels``.

    This is the arm selector: the Fibonacci arm keeps swings that retraced near 0.236/0.382/0.5/
    0.618; the placebo arm keeps swings that retraced near the arbitrary fractions. Same machinery,
    different level set — the head-to-head the study is about.
    """
    if retr_df.empty:
        return retr_df
    mask = retr_df["retr"].apply(lambda r: _near_any(r, levels, tol))
    return retr_df.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fixed-horizon forward-return backtest
# --------------------------------------------------------------------------- #
def run_trades(close: np.ndarray, touches: pd.DataFrame, horizon: int = 20,
               cost_bps: float = 1.0, directions: np.ndarray | None = None) -> pd.DataFrame:
    """Run fixed-horizon reversal trades from signal bars; return a per-trade ledger.

    For each signal bar ``sig_idx`` (already public) enter at the **next** bar's close (one-day
    execution lag) and exit ``horizon`` trading days later. ``directions`` overrides the sign
    vector (the coin placebo passes +/-1 aligned to the rows). ``cost_bps`` is one-way; a round
    trip pays it twice.

    Columns: ``entry_idx, dir, ret_gross, ret_net``.
    """
    n = len(close)
    if len(touches) == 0:
        return pd.DataFrame(columns=["entry_idx", "dir", "ret_gross", "ret_net"])
    dirs = (np.asarray(directions, dtype=int) if directions is not None
            else touches["dir"].to_numpy(dtype=int))
    rows = []
    for sig, d in zip(touches["sig_idx"].to_numpy(dtype=int), dirs):
        e = sig + 1
        x = e + horizon
        if e >= n or x >= n:
            continue
        gross = d * (close[x] / close[e] - 1.0)
        rows.append({"entry_idx": e, "dir": int(d),
                     "ret_gross": gross, "ret_net": gross - 2.0 * cost_bps * 1e-4})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return float(sample.mean() / se) if se > 0 else float("nan")


def hac_t(sample: np.ndarray) -> float:
    """Newey-West (HAC) t-stat on the mean of ``sample`` (no external dependency)."""
    r = np.asarray(sample, dtype=float)
    r = r[np.isfinite(r)]
    nn = r.size
    if nn <= 5:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (nn / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / nn
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / nn
    se = np.sqrt(max(lrv, 0.0) / nn)
    return float(mu / se) if se > 0 else float("nan")


def two_sample_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch two-sample t of mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger."""
    if len(ledger) == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "mean_bps": float("nan"),
                "sharpe": float("nan"), "t": float("nan"), "hac_t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    nn = r.size
    return {
        "n_trades": int(nn),
        "win_rate": float((r > 0).mean()) if nn else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if nn else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if nn > 1 and r.std() > 0 else float("nan"),
        "t": ttest_vs_zero(r),
        "hac_t": hac_t(r),
    }


def coin_pvalue(close: np.ndarray, touches: pd.DataFrame, horizon: int = 20,
                n_draws: int = 5000, seed: int = 541) -> dict:
    """Same-bars random-direction placebo: does betting reversal beat a coin at the same touches?

    Keep the same touch bars, randomise the +/-1 direction with a fair coin ``n_draws`` times.
    ``p`` = P[shuffled mean gross forward return >= observed]. Run on gross returns so the placebo
    measures pure direction.
    """
    led = run_trades(close, touches, horizon, cost_bps=0.0)
    if len(led) == 0:
        return {"obs": float("nan"), "p_value": float("nan")}
    obs = float(led["ret_gross"].mean())
    mags = led["ret_gross"].to_numpy() * led["dir"].to_numpy()   # unsigned move
    rng = np.random.default_rng(seed)
    draws = np.array([float((rng.choice([-1, 1], size=len(mags)) * mags).mean())
                      for _ in range(n_draws)])
    return {"obs": obs, "p_value": float((draws >= obs).mean())}


# --------------------------------------------------------------------------- #
# The headline experiment — Fib arm vs placebo arm
# --------------------------------------------------------------------------- #
def run_arm(close: np.ndarray, retr_df: pd.DataFrame, levels, horizon: int,
            cost_bps: float, tol: float, coin_draws: int, seed: int) -> dict:
    """One arm (a set of retracement levels): bin swings -> backtest + inference + coin placebo."""
    touches = bin_touches(retr_df, levels, tol=tol)
    led = run_trades(close, touches, horizon, cost_bps=cost_bps)
    net = summarize(led, "ret_net")
    gross = summarize(led, "ret_gross")
    coin = coin_pvalue(close, touches, horizon, n_draws=coin_draws, seed=seed)
    return {"n_touches": int(len(touches)), "net": net, "gross": gross,
            "coin_p": coin["p_value"], "ledger": led}


def run_experiment(bars: pd.DataFrame, pct: float = 0.05, horizon: int = 20,
                   cost_bps: float = 1.0, tol: float = 0.025,
                   fib_levels=FIB_LEVELS, placebo_levels=PLACEBO_LEVELS,
                   coin_draws: int = 5000, seed: int = 541) -> dict:
    """Orchestrator for one tape/horizon: Fib arm vs placebo arm + the Fib-minus-placebo edge.

    Returns the headline dict: each arm's reversal stats and coin p-value, and the two-sample
    *t* of the Fib arm's gross forward returns minus the placebo arm's (the decisive number — are
    Fibonacci levels *special*?).
    """
    piv = zigzag(bars["close"].to_numpy(dtype=float), pct=pct)
    close = bars["close"].to_numpy(dtype=float)
    retr_df = swing_retracements(piv)
    fib = run_arm(close, retr_df, fib_levels, horizon, cost_bps, tol, coin_draws, seed)
    plc = run_arm(close, retr_df, placebo_levels, horizon, cost_bps, tol, coin_draws, seed + 1)
    edge_t = two_sample_t(fib["ledger"]["ret_gross"].to_numpy(),
                          plc["ledger"]["ret_gross"].to_numpy())
    fib_mean_bps = fib["gross"]["mean_bps"]
    plc_mean_bps = plc["gross"]["mean_bps"]
    return {
        "n_pivots": int(len(piv)),
        "fib": {k: v for k, v in fib.items() if k != "ledger"},
        "placebo": {k: v for k, v in plc.items() if k != "ledger"},
        "edge_bps": (fib_mean_bps - plc_mean_bps
                     if np.isfinite(fib_mean_bps) and np.isfinite(plc_mean_bps) else float("nan")),
        "edge_t": edge_t,
        "pct": pct, "horizon": horizon, "cost_bps": cost_bps, "tol": tol,
    }


# --------------------------------------------------------------------------- #
# Arm split on a swing panel (retr, dir, forward_ret) — used by the control
# --------------------------------------------------------------------------- #
def arm_forward(swings: pd.DataFrame, levels, tol: float = 0.025) -> np.ndarray:
    """Direction-signed forward returns of the swings whose retracement is near ``levels``."""
    sub = bin_touches(swings, levels, tol=tol)
    if sub.empty:
        return np.array([])
    return (sub["dir"].to_numpy(dtype=float) * sub["forward_ret"].to_numpy(dtype=float))


def swing_panel_edge(swings: pd.DataFrame, fib_levels=FIB_LEVELS,
                     placebo_levels=PLACEBO_LEVELS, tol: float = 0.025) -> dict:
    """Fib-arm *t* vs 0, placebo-arm *t*, and the Fib-minus-placebo two-sample edge *t*."""
    fib = arm_forward(swings, fib_levels, tol)
    plc = arm_forward(swings, placebo_levels, tol)
    return {
        "fib_t": ttest_vs_zero(fib), "placebo_t": ttest_vs_zero(plc),
        "fib_mean_bps": float(fib.mean() * 1e4) if fib.size else float("nan"),
        "placebo_mean_bps": float(plc.mean() * 1e4) if plc.size else float("nan"),
        "edge_t": two_sample_t(fib, plc),
        "n_fib": int(fib.size), "n_placebo": int(plc.size),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control — seed-robust (house rule: >= 20 seeds)
# --------------------------------------------------------------------------- #
def synthetic_control(data_mod, fib_pull: float, n_seeds: int = 25, base_seed: int = 541,
                      n: int = 1500) -> dict:
    """Average the Fib-minus-placebo edge *t* over ``n_seeds`` synthetic swing panels.

    House rule: any synthetic-dependent claim averages the statistic over >= 20 seeds so no single
    lucky RNG seed manufactures significance. Uses ``data_mod.synthetic_swings`` — the clean swing
    panel with a planted level-effect. ``fib_pull = 0`` is the null (edge *t* ≈ 0); ``fib_pull > 0``
    plants a Fibonacci reversal edge the binning + two-sample test must recover (edge *t* > 0,
    growing with ``fib_pull``).

    Returns the mean Fib-arm *t*, the mean Fib-minus-placebo edge *t*, and mean bin sizes.
    """
    fib_ts, edge_ts, n_fib = [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        swings, _ = data_mod.synthetic_swings(n=n, fib_pull=fib_pull, seed=s)
        res = swing_panel_edge(swings)
        fib_ts.append(res["fib_t"])
        edge_ts.append(res["edge_t"])
        n_fib.append(res["n_fib"])
    return {
        "fib_pull": fib_pull,
        "mean_fib_t": float(np.nanmean(fib_ts)),
        "mean_edge_t": float(np.nanmean(edge_ts)),
        "mean_n_fib": float(np.nanmean(n_fib)),
        "n_seeds": n_seeds,
    }
