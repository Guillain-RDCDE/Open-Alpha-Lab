"""Strategy + inference for Study 352 — Opening-Range Breakout (ORB).

The viral pitch (e.g. day-trading TikTok, and a *real* steelman in Zarattini & Grewal 2023):
mark the high/low of the first **R** minutes after the open (the "opening range"). If price
**breaks above** the range high, go **long**; if it breaks **below** the low, go **short**.
Hold to the close (one bar later than the breakout — one execution lag). The claim is that the
first break of the morning range carries genuine directional information.

The honest test is **not** "does ORB make money" — in a 60-day bull sample, *any* mostly-long
intraday rule makes money. The test is: **does ORB beat a same-exposure RANDOM entry on the
same days?** A random control that enters at a random bar, in a random (or drift-matched)
direction, and holds the *same fraction of the session* strips out the day's drift and leaves
only the breakout's marginal information. We then charge realistic one-way intraday costs
(spread + slippage, in bps) on the NAV and see whether anything survives.

Returns are simple per-day returns (entry price → close), one execution lag applied once
(enter on the bar *after* the breakout bar). Pure numpy.
"""

from __future__ import annotations

import math

import numpy as np

# per-bar OHLC column indices in a days array (n_days, bars, 4)
O, H, L, C = 0, 1, 2, 3

# A 5-minute regular session: 78 bars. First 5 min = 1 bar; first 15 min = 3 bars.
BAR_MINUTES = 5


def _orb_bars(opening_range_min: int) -> int:
    return max(1, opening_range_min // BAR_MINUTES)


def orb_trades(days: np.ndarray, opening_range_min: int = 5,
               lag: int = 1) -> dict:
    """Run the opening-range-breakout rule on every day.

    For each session: the opening range is the high/low of the first ``opening_range_min``
    minutes. We scan the *remaining* bars for the first close that breaks the range; the trade
    is entered ``lag`` bars later (the open of that bar — one execution lag) in the breakout's
    direction, and exits at the session close. Days with no break sit flat (return 0).

    Returns a dict with the per-day signed returns ``ret`` (np.ndarray, one per day; 0 if no
    trade), a boolean ``traded`` mask, the entry bar index ``entry_bar`` (or -1), the trade
    ``direction`` (+1 long / -1 short / 0), and the held fraction ``hold_frac`` of the session
    (entry→close bars / total) — the exposure the random null must match.
    """
    n, bars, _ = days.shape
    k = _orb_bars(opening_range_min)
    rng_hi = days[:, :k, H].max(axis=1)
    rng_lo = days[:, :k, L].min(axis=1)

    ret = np.zeros(n)
    traded = np.zeros(n, dtype=bool)
    entry_bar = np.full(n, -1, dtype=int)
    direction = np.zeros(n, dtype=int)
    hold_frac = np.zeros(n)

    for i in range(n):
        closes = days[i, :, C]
        for b in range(k, bars - lag):
            up = closes[b] > rng_hi[i]
            dn = closes[b] < rng_lo[i]
            if not (up or dn):
                continue
            d = 1 if up else -1
            eb = b + lag
            entry_px = days[i, eb, O]
            exit_px = days[i, bars - 1, C]
            ret[i] = d * (exit_px / entry_px - 1.0)
            traded[i] = True
            entry_bar[i] = eb
            direction[i] = d
            hold_frac[i] = (bars - 1 - eb) / bars
            break

    return {"ret": ret, "traded": traded, "entry_bar": entry_bar,
            "direction": direction, "hold_frac": hold_frac,
            "n_days": n, "bars": bars, "k": k}


def random_null(days: np.ndarray, trades: dict, mode: str = "matched_dir",
                n_iter: int = 2000, seed: int = 352) -> dict:
    """Same-exposure random-entry control, matched day-by-day to the ORB trades.

    On each ORB-traded day we draw a random entry bar in the *same window* the breakout could
    have fired (after the opening range, before the last ``lag`` bars), hold to the close, and
    take a direction:

    * ``mode='matched_dir'`` — same direction as the ORB trade that day (isolates *timing*:
      does breaking the range beat entering at a random time, holding the same way?).
    * ``mode='coin'`` — a fair coin direction (isolates timing **and** sign).
    * ``mode='drift'`` — always long (the day's drift benchmark; in a bull tape this is the
      hardest control to beat, and the most honest).

    Returns the mean per-day return of the null across ``n_iter`` resamples, its standard
    error, and the full distribution of null *means* (for a permutation-style comparison).
    """
    rng = np.random.default_rng(seed)
    n, bars, _ = days.shape
    k = trades["k"]
    traded = trades["traded"]
    idx = np.where(traded)[0]
    if idx.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "dist": np.array([])}

    closes = days[:, bars - 1, C]
    opens = days[:, :, O]
    means = np.empty(n_iter)
    for it in range(n_iter):
        day_ret = np.empty(idx.size)
        for j, i in enumerate(idx):
            lo_b, hi_b = k + 1, bars - 1
            eb = int(rng.integers(lo_b, hi_b))
            if mode == "matched_dir":
                d = trades["direction"][i]
            elif mode == "coin":
                d = 1 if rng.random() < 0.5 else -1
            else:  # drift = always long
                d = 1
            day_ret[j] = d * (closes[i] / opens[i, eb] - 1.0)
        means[it] = day_ret.mean()
    return {"mean": float(means.mean()), "se": float(means.std(ddof=1)),
            "dist": means, "n": int(idx.size)}


def _tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(x.size)))


def evaluate(days: np.ndarray, opening_range_min: int = 5, cost_bps: float = 0.0,
             null_mode: str = "matched_dir", n_iter: int = 2000,
             seed: int = 352) -> dict:
    """Headline ORB-vs-null evaluation on a set of days, net of ``cost_bps`` one-way.

    The cost is charged once per round trip (entry + exit = ``2 * cost_bps`` on NAV) on every
    traded day. The Signal-axis statistic is the **paired** t-stat of (ORB per-day return −
    matched random per-day return): a robust t ≥ 2 means breaking the range adds information a
    random same-exposure entry does not. We also report the ORB mean, hit-rate, and the
    permutation p-value of the ORB mean against the null distribution of means.
    """
    tr = orb_trades(days, opening_range_min=opening_range_min)
    rt = tr["ret"].copy()
    traded = tr["traded"]
    round_cost = 2.0 * cost_bps / 1e4
    rt_net = rt - traded * round_cost

    nl = random_null(days, tr, mode=null_mode, n_iter=n_iter, seed=seed)
    idx = np.where(traded)[0]

    # paired ORB - random, day by day, using one representative random draw per day (seeded)
    rng = np.random.default_rng(seed + 1)
    bars = tr["bars"]
    k = tr["k"]
    closes = days[:, bars - 1, C]
    paired = np.empty(idx.size)
    for j, i in enumerate(idx):
        eb = int(rng.integers(k + 1, bars - 1))
        if null_mode == "matched_dir":
            d = tr["direction"][i]
        elif null_mode == "coin":
            d = 1 if rng.random() < 0.5 else -1
        else:
            d = 1
        rnd = d * (closes[i] / days[i, eb, O] - 1.0)
        paired[j] = rt_net[i] - rnd

    orb_traded = rt_net[traded]
    t_vs_zero = _tstat(orb_traded)
    t_paired = _tstat(paired)
    orb_mean = float(orb_traded.mean()) if idx.size else float("nan")
    # permutation p: P(null mean >= ORB net mean)
    perm_p = (float((nl["dist"] >= orb_mean).mean())
              if nl["dist"].size else float("nan"))
    return {
        "n_days": tr["n_days"],
        "n_trades": int(traded.sum()),
        "trade_rate": float(traded.mean()),
        "orb_mean_bps": orb_mean * 1e4,
        "orb_hit": float((orb_traded > 0).mean()) if idx.size else float("nan"),
        "orb_sharpe_day": (orb_mean / orb_traded.std(ddof=1)) if idx.size > 1 else float("nan"),
        "null_mean_bps": nl["mean"] * 1e4,
        "null_se_bps": nl["se"] * 1e4,
        "edge_bps": (orb_mean - nl["mean"]) * 1e4,
        "t_vs_zero": t_vs_zero,
        "t_paired": t_paired,
        "perm_p": perm_p,
        "cost_bps": cost_bps,
        "mean_hold_frac": float(tr["hold_frac"][traded].mean()) if idx.size else float("nan"),
    }


def cost_sweep(days: np.ndarray, opening_range_min: int = 5,
               costs=(0.0, 1.0, 2.0, 3.0, 5.0), null_mode: str = "matched_dir",
               seed: int = 352) -> list[dict]:
    """ORB net edge across a grid of one-way costs (bps). Each dict gains nothing extra."""
    return [evaluate(days, opening_range_min=opening_range_min, cost_bps=c,
                     null_mode=null_mode, seed=seed) for c in costs]


def breakeven_cost(days: np.ndarray, opening_range_min: int = 5,
                   null_mode: str = "matched_dir", seed: int = 352) -> float:
    """One-way cost (bps) at which the ORB net edge over the null crosses zero.

    Solves ``edge_bps(c) = 0``. Since the net ORB mean falls by ``2*c`` bps per bp of cost and
    the null is cost-free, the break-even is simply ``gross_edge / 2`` (in bps), but we return
    it via the actual evaluator so it always matches the reported numbers.
    """
    gross = evaluate(days, opening_range_min=opening_range_min, cost_bps=0.0,
                     null_mode=null_mode, seed=seed)["edge_bps"]
    return gross / 2.0
