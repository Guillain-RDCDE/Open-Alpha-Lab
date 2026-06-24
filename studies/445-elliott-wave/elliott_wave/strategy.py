"""The mechanical Elliott-Wave proxy + its honest controls — Study 445.

Elliott Wave Theory says price unfolds in a fractal **five-wave impulse** (1-2-3-4-5) in the
trend direction, then a **three-wave correction** (A-B-C). Proponents add Fibonacci rules:
wave 2 retraces ~50–61.8% of wave 1, wave 3 is the strongest and longest leg (the one every
Elliottician wants to be long for), and a completed five-wave impulse is followed by a
correction.

The theory is **irreducibly subjective** — the "correct" count is only knowable in hindsight,
and two analysts label the same chart differently. There is no single falsifiable rule. So we
test the **tightest mechanical version proponents accept**:

1. **ZigZag swing detector** (the standard EW labeling tool): mark a new pivot only after price
   reverses by at least ``pct`` from the last extreme. This produces an alternating
   high/low/high/low pivot sequence — the raw material every wave count is built on.

2. **The wave-3 prediction (the tradable claim).** Take a candidate impulse 1-2: a swing leg
   (wave 1) followed by a pullback (wave 2) that retraces a **Fibonacci** fraction of wave 1
   (between ``fib_lo`` and ``fib_hi``, default 38.2%–78.6%, centred on the textbook 50–61.8%).
   At the wave-2 pivot, the theory says go **with the wave-1 direction** — wave 3 will extend.
   We enter **one day after** the pivot is confirmed (no look-ahead) and hold ``horizon`` days.

3. **The correction prediction (completed-impulse claim).** After a full five-pivot impulse, the
   theory predicts a corrective move *against* the impulse direction.

Inference — the desk's shared spirit:

  * a **one-sample t** (and a HAC/Newey-West t) of the directional forward return vs 0;
  * a **label-shuffle placebo**: keep the *same* entry bars, randomise the direction with a fair
    coin, many times — the honest answer to "does the wave count add directional information
    over a coin at the same swing points?";
  * a **win-rate** vs the 50% base rate;
  * **one-day execution lag** and **one-way costs × turnover**.

No look-ahead: a pivot at bar p is only *confirmed* once price has reversed ``pct`` from it,
which happens at some bar q > p; we enter at q+1's close. The detector returns the confirmation
bar so the entry is always strictly after the information is public.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)


# --------------------------------------------------------------------------- #
# ZigZag swing detector (the EW labeling tool)
# --------------------------------------------------------------------------- #
def zigzag(close: np.ndarray, pct: float = 0.05) -> pd.DataFrame:
    """Percentage ZigZag pivots on a close series.

    Walk forward tracking the running extreme since the last confirmed pivot. A new pivot is
    *confirmed* the first time price reverses by at least ``pct`` (e.g. 0.05 = 5%) from that
    extreme; the pivot is stamped at the bar of the extreme (``piv_idx``) but only *known* at
    the reversal bar (``conf_idx``) — the look-ahead-free confirmation bar.

    Returns a DataFrame of alternating pivots with columns:
      ``piv_idx`` (bar of the extreme), ``conf_idx`` (bar it was confirmed = known),
      ``price`` (extreme price), ``kind`` (+1 = swing high, -1 = swing low).
    """
    n = len(close)
    if n < 3:
        return pd.DataFrame(columns=["piv_idx", "conf_idx", "price", "kind"])

    pivots = []
    # Running extremes since the last confirmed pivot, tracked in both directions until we
    # learn which way price first breaks by `pct`.
    hi_idx = lo_idx = 0
    hi_px = lo_px = close[0]
    direction = 0  # 0 = undetermined, +1 = uptrend (seeking next high), -1 = downtrend

    for i in range(1, n):
        px = close[i]
        if direction >= 0 and px > hi_px:
            hi_px, hi_idx = px, i
        if direction <= 0 and px < lo_px:
            lo_px, lo_idx = px, i

        if direction == 0:
            # decide the initial direction on the first qualifying break
            if px <= hi_px * (1.0 - pct):
                pivots.append((hi_idx, i, hi_px, +1))   # confirmed a swing high
                direction = -1
                lo_px, lo_idx = px, i
            elif px >= lo_px * (1.0 + pct):
                pivots.append((lo_idx, i, lo_px, -1))   # confirmed a swing low
                direction = +1
                hi_px, hi_idx = px, i
        elif direction > 0:
            # in an uptrend: confirm the swing high when price reverses down by pct
            if px <= hi_px * (1.0 - pct):
                pivots.append((hi_idx, i, hi_px, +1))
                direction = -1
                lo_px, lo_idx = px, i
        else:
            # in a downtrend: confirm the swing low when price reverses up by pct
            if px >= lo_px * (1.0 + pct):
                pivots.append((lo_idx, i, lo_px, -1))
                direction = +1
                hi_px, hi_idx = px, i

    piv = pd.DataFrame(pivots, columns=["piv_idx", "conf_idx", "price", "kind"])
    return piv.drop_duplicates(subset=["piv_idx"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Elliott wave-2 -> wave-3 entries (the tradable prediction)
# --------------------------------------------------------------------------- #
def wave3_entries(pivots: pd.DataFrame, fib_lo: float = 0.382,
                  fib_hi: float = 0.786) -> pd.DataFrame:
    """Candidate 'buy wave 3' entries from a ZigZag pivot sequence.

    Slide a window of three consecutive pivots P0, P1, P2 (the prospective wave-1 start,
    wave-1 end, wave-2 end). For a valid impulse 1-2:
      * P0->P1 is wave 1 (a swing in the impulse direction),
      * P1->P2 is wave 2 (a pullback that does NOT exceed the start of wave 1, an EW rule), and
      * the wave-2 retracement of wave 1 lies in the Fibonacci band [fib_lo, fib_hi]
        (the textbook 50–61.8% sits inside the default 38.2–78.6% band).
    The theory predicts wave 3 then extends in the wave-1 direction. We take the **direction**
    of wave 1 as the trade direction, and the **confirmation bar** of P2 as the signal bar
    (entry is one day later, in the backtest).

    Returns one row per entry: ``sig_idx`` (P2 confirmation bar = known), ``dir`` (+1 up / -1
    down impulse), ``retr`` (the wave-2 retracement fraction).
    """
    out = []
    for k in range(2, len(pivots)):
        p0 = pivots.iloc[k - 2]
        p1 = pivots.iloc[k - 1]
        p2 = pivots.iloc[k]
        # wave 1 direction: from P0 to P1
        w1 = p1["price"] - p0["price"]
        if w1 == 0:
            continue
        direction = 1 if w1 > 0 else -1
        # wave 2 must be a pullback of wave 1 (opposite sign), not exceeding P0 (EW rule 2)
        w2 = p2["price"] - p1["price"]
        if np.sign(w2) == np.sign(w1):
            continue
        retr = abs(w2) / abs(w1)
        if direction > 0 and p2["price"] <= p0["price"]:
            continue  # wave 2 broke below wave-1 start -> not a valid impulse
        if direction < 0 and p2["price"] >= p0["price"]:
            continue
        if not (fib_lo <= retr <= fib_hi):
            continue
        out.append({"sig_idx": int(p2["conf_idx"]), "dir": int(direction),
                    "retr": float(retr)})
    return pd.DataFrame(out)


def completed_impulse_entries(pivots: pd.DataFrame) -> pd.DataFrame:
    """'Sell the completed five-wave impulse' entries — the correction prediction.

    After five alternating pivots forming an impulse (1-2-3-4-5) in one direction, the theory
    predicts a corrective A-B-C *against* the impulse. We flag the 5th pivot's confirmation bar
    and bet **against** the impulse direction. The mechanical impulse test is light (five
    alternating pivots with the 5th beyond the 3rd beyond the 1st in the trend direction) — the
    tightest version we can encode without subjective relabelling.
    """
    out = []
    for k in range(4, len(pivots)):
        seg = pivots.iloc[k - 4:k + 1]
        prices = seg["price"].values
        # net direction of the five-pivot move
        direction = 1 if prices[-1] > prices[0] else -1
        # crude monotone-impulse check: extremes progress in the trend direction
        if direction > 0 and not (prices[4] > prices[2] > prices[0]):
            continue
        if direction < 0 and not (prices[4] < prices[2] < prices[0]):
            continue
        out.append({"sig_idx": int(seg.iloc[-1]["conf_idx"]), "dir": int(-direction)})
    return pd.DataFrame(out)


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of +/-1 — the placebo arm's coin (same entry bars)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# --------------------------------------------------------------------------- #
# Fixed-horizon forward-return backtest
# --------------------------------------------------------------------------- #
def run_trades(bars: pd.DataFrame, entries: pd.DataFrame, horizon: int = 20,
               cost_bps: float = 1.0, directions: np.ndarray | None = None) -> pd.DataFrame:
    """Run fixed-horizon directional trades and return a per-trade ledger.

    For each signal bar ``sig_idx`` (the pivot confirmation bar, already public), enter at the
    **next** bar's close (one-day execution lag) and exit ``horizon`` trading days later (or the
    last available bar). ``directions`` overrides the entry sign vector (the placebo passes a
    +/-1 coin aligned to ``entries``). ``cost_bps`` is a one-way cost; a round trip pays it
    twice, deducted from the gross return.

    Columns: ``entry_idx, dir, ret_gross, ret_net``.
    """
    close = bars["close"].to_numpy(dtype=float)
    n = len(close)
    if len(entries) == 0:
        return pd.DataFrame(columns=["entry_idx", "dir", "ret_gross", "ret_net"])
    dirs = (np.asarray(directions, dtype=int) if directions is not None
            else entries["dir"].to_numpy(dtype=int))
    rows = []
    for sig, d in zip(entries["sig_idx"].to_numpy(dtype=int), dirs):
        e = sig + 1                       # enter one day after the signal is known
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
    """Newey-West (HAC) t-stat on the mean of ``sample`` (no quantlab dependency)."""
    r = np.asarray(sample, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n <= 5:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger."""
    if len(ledger) == 0:
        return {k: float("nan") for k in
                ["n_trades", "win_rate", "mean_bps", "sharpe", "t", "hac_t"]} | {"n_trades": 0}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": ttest_vs_zero(r),
        "hac_t": hac_t(r),
    }


def placebo_pvalue(bars: pd.DataFrame, entries: pd.DataFrame, horizon: int = 20,
                   n_draws: int = 5000, seed: int = 445, cost_bps: float = 0.0) -> dict:
    """Label-shuffle placebo: same entry bars, random +/-1 direction, many times.

    ``p`` = P[shuffled mean forward return >= observed]. The honest test for whether the wave
    count adds *directional* information over a coin flip placed at the very same swing points.
    Run on gross returns (cost_bps=0) so the placebo measures pure direction.
    """
    led = run_trades(bars, entries, horizon, cost_bps=cost_bps)
    if len(led) == 0:
        return {"obs": float("nan"), "p_value": float("nan"), "draws": np.array([])}
    obs = float(led["ret_gross"].mean())
    # the gross magnitude of each trade, signed by the wave direction
    base = run_trades(bars, entries, horizon, cost_bps=cost_bps)
    mags = (base["ret_gross"].to_numpy() * base["dir"].to_numpy())  # un-signed move
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        d = rng.choice([-1, 1], size=len(mags))
        draws[i] = float((d * mags).mean())
    p = float((draws >= obs).mean())
    return {"obs": obs, "p_value": p, "draws": draws}


def run_experiment(bars: pd.DataFrame, pct: float = 0.05, horizon: int = 20,
                   cost_bps: float = 1.0, fib_lo: float = 0.382, fib_hi: float = 0.786,
                   n_draws: int = 5000, seed: int = 445) -> dict:
    """Orchestrator: ZigZag -> wave-3 entries -> backtest + inference + placebo + coin control.

    Returns the headline dict for one tape/horizon: the wave-3 long/short stats, the
    same-bars random-direction coin control, and the label-shuffle placebo p-value.
    """
    piv = zigzag(bars["close"].to_numpy(dtype=float), pct=pct)
    ent = wave3_entries(piv, fib_lo=fib_lo, fib_hi=fib_hi)
    led = run_trades(bars, ent, horizon, cost_bps=cost_bps)
    wave = summarize(led, "ret_net")
    wave_gross = summarize(led, "ret_gross")
    # coin control: same bars, random direction
    coin_led = run_trades(bars, ent, horizon, cost_bps=cost_bps,
                          directions=random_directions(len(ent), seed=seed))
    coin = summarize(coin_led, "ret_gross")
    plac = placebo_pvalue(bars, ent, horizon, n_draws=n_draws, seed=seed)
    return {
        "n_pivots": int(len(piv)), "n_entries": int(len(ent)),
        "wave": wave, "wave_gross": wave_gross, "coin": coin,
        "placebo_p": plac["p_value"], "placebo_obs": plac["obs"],
        "pct": pct, "horizon": horizon, "fib_lo": fib_lo, "fib_hi": fib_hi,
    }
