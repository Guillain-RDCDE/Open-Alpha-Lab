"""Event-study engine + honest controls — Study 845 (Stadium Naming-Rights Curse).

The claim under test, steelmanned (managerial-hubris / peak-earnings signaling): a
company that pays up for expensive stadium naming rights is, on average, a company at
or near a self-confident peak — spending shareholder cash on a vanity trophy rather
than on its core business — and such companies subsequently **underperform**. The
folklore's evidence is a short list of vivid blow-ups: Enron Field (bankrupt 2001), the
FTX Arena (collapsed 2022), the MCI Center (WorldCom fraud). The honest question is
whether that is a **systematic** effect across dated, publicly-traded deals or a
**cherry-picked** handful of survivors of hindsight.

The machinery, one execution lag documented throughout:

* ``forward_bhar`` — the sponsor's **buy-and-hold abnormal return** over the
  ``window`` sessions after the deal: (sponsor total return) − (SPY total return),
  entering at the close of the first NYSE session on/after the announcement date
  (``searchsorted`` snap — the single documented execution lag: the deal is public by
  that close, so zero look-ahead). BHAR is the standard long-horizon event-study
  statistic (Barber & Lyon 1997) — cleaner than summing daily abnormal returns over a
  one-to-two-year horizon.
* ``stack_bhar`` — the cross-section of per-deal BHARs at a given horizon, dropping
  deals whose window runs off the tape or whose ticker has no coverage (named, not
  zero-filled).
* ``car_stats`` — cross-event mean BHAR, one-sample *t* (deals treated as independent
  cross-sectional observations), a hit rate with a Wilson interval, and a
  ``newey_west_t`` cross-check against calendar-clustering (several deals in the same
  year share market weather).
* ``era_split`` — the sub-era robustness a "Real" stamp on this desk requires: does any
  effect hold before *and* after 2010, or is it one era (or a couple of names)?
* ``placebo_pvalue`` — the falsification control: keep the same tickers but read each
  deal's BHAR from a **random pseudo-announcement date** on that ticker's own tape,
  thousands of times — the observed cross-event mean must sit in the tail of that null.
* ``curse_overlay`` — the tradable read of the "curse": **short the sponsor, long SPY**
  for ``window`` sessions after each deal, costed one-way × NAV per leg plus borrow on
  the short. A shorting strategy, so borrow is charged; almost always Mirage/Fragile.

N deals is small → low power → this study expects **None** unless a large, era-robust
effect is actually there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (canonical desk set — mirrors 803)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> tuple[float, float]:
    """Mean and one-sample t-stat of ``x`` (observations treated as independent)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 or x.std(ddof=1) == 0:
        return (float(np.nan if n == 0 else x.mean()), float("nan"))
    se = x.std(ddof=1) / np.sqrt(n)
    return (float(x.mean()), float(x.mean() / se))


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 4) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0.

    Applied here to the time-ordered per-deal BHARs as a robustness cross-check: deals
    clustered in the same calendar era share market weather, so a HAC standard error is
    a conservative alternative to the plain cross-sectional one-sample t.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson (1927) score interval for a proportion ``k/n``."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Buy-and-hold abnormal return around a single deal
# --------------------------------------------------------------------------- #
def forward_bhar(sponsor: pd.Series, spy: pd.Series, event_date, window: int = 252
                 ) -> float | None:
    """Buy-and-hold abnormal return over ``[+0..+window]`` after the deal.

    Enter at the close of the first session on/after ``event_date`` (searchsorted snap),
    hold ``window`` sessions. BHAR = (sponsor close ratio − 1) − (SPY close ratio − 1)
    over that window, both measured on the sponsor's own trading calendar (SPY is
    reindexed onto the sponsor's dates first, forward-filled across any calendar gaps).
    Returns ``None`` if the ticker has no observation on/after the date or the window
    runs off the end of its tape.
    """
    s = sponsor.dropna()
    if s.empty:
        return None
    idx = s.index
    pos = idx.searchsorted(pd.Timestamp(event_date))
    if pos >= len(idx):
        return None
    exit_ = pos + window
    if exit_ >= len(idx):
        return None
    spy_al = spy.reindex(idx).ffill()
    p0, p1 = s.iat[pos], s.iat[exit_]
    m0, m1 = spy_al.iat[pos], spy_al.iat[exit_]
    if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(m0) and np.isfinite(m1)) \
            or p0 <= 0 or m0 <= 0:
        return None
    return float((p1 / p0 - 1.0) - (m1 / m0 - 1.0))


def stack_bhar(spy: pd.Series, prices: dict[str, pd.Series], deals: pd.DataFrame,
               window: int = 252) -> tuple[np.ndarray, pd.DataFrame]:
    """Cross-section of per-deal BHARs at ``window``.

    ``deals`` is a frame with ``date`` and ``ticker`` columns. Deals whose ticker has
    no cached tape, or whose window runs off the tape, are dropped (named honestly via
    the returned kept-frame, never zero-filled). Returns ``(bhar_array, kept_frame)``
    where ``kept_frame`` carries the deal's date/ticker/venue and its ``bhar``.
    """
    rows = []
    for _, r in deals.iterrows():
        t = r["ticker"]
        if t not in prices:
            continue
        b = forward_bhar(prices[t], spy, r["date"], window)
        if b is None:
            continue
        rows.append({"date": r["date"], "ticker": t,
                     "venue": r.get("venue", ""), "bhar": b})
    kept = pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows \
        else pd.DataFrame(columns=["date", "ticker", "venue", "bhar"])
    return kept["bhar"].to_numpy(dtype=float), kept


# --------------------------------------------------------------------------- #
# Headline cross-event stats
# --------------------------------------------------------------------------- #
def car_stats(spy: pd.Series, prices: dict[str, pd.Series], deals: pd.DataFrame,
              window: int = 252, nw_lags: int = 4) -> dict:
    """Cross-event mean BHAR + one-sample t + NW t + Wilson hit rate at ``window``.

    The claim (a *curse*) predicts a **negative** mean BHAR. ``hit`` counts deals with
    negative BHAR (sponsor underperformed SPY). ``t`` is the primary cross-sectional
    one-sample t; ``t_nw`` is the calendar-clustering-robust cross-check.
    """
    bhar, kept = stack_bhar(spy, prices, deals, window)
    n = bhar.size
    if n == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan"), "t_nw": float("nan"),
                "hit": 0, "hit_rate": float("nan"), "wilson": (float("nan"), float("nan")),
                "kept": kept}
    mean, t = one_sample_t(bhar)
    t_nw = newey_west_t(bhar, lags=nw_lags)
    hit = int((bhar < 0).sum())
    return {"n": n, "mean": mean, "t": t, "t_nw": t_nw,
            "hit": hit, "hit_rate": hit / n, "wilson": wilson_interval(hit, n),
            "median": float(np.median(bhar)), "kept": kept}


def era_split(spy: pd.Series, prices: dict[str, pd.Series], deals: pd.DataFrame,
              window: int = 252, cut: str = "2010-01-01") -> dict:
    """Sub-era robustness: mean BHAR + one-sample t before and after ``cut``.

    A "Real" stamp on this desk requires an effect that holds in *both* halves, not one
    driven by a single era (or a couple of names). Returns a dict per era.
    """
    _, kept = stack_bhar(spy, prices, deals, window)
    out = {}
    for name, mask in (("pre", kept["date"] < cut), ("post", kept["date"] >= cut)):
        x = kept.loc[mask, "bhar"].to_numpy(dtype=float)
        mean, t = one_sample_t(x)
        out[name] = {"n": int(np.isfinite(x).sum()), "mean": mean, "t": t}
    return out


# --------------------------------------------------------------------------- #
# Falsification — random pseudo-announcement dates on the same tickers
# --------------------------------------------------------------------------- #
def placebo_pvalue(spy: pd.Series, prices: dict[str, pd.Series], deals: pd.DataFrame,
                   window: int = 252, n_draws: int = 3000, seed: int = 845) -> dict:
    """Random-date placebo: keep the same tickers but read each deal's BHAR from a
    random pseudo-announcement date on that ticker's own tape, ``n_draws`` times.

    Breaks the deal→outcome link while preserving each name's own return distribution
    and the sample size. A real curse must put the observed cross-event mean in the
    **left** tail (more negative than random entry into the same names). Vectorised:
    for each ticker we precompute the full vector of window BHARs at every eligible
    entry position, then each draw samples one position per kept ticker.
    """
    _, kept = stack_bhar(spy, prices, deals, window)
    obs = float(np.nanmean(kept["bhar"].to_numpy(dtype=float))) if len(kept) else float("nan")
    if len(kept) == 0:
        return {"obs": obs, "placebo_mean": float("nan"), "placebo_sd": float("nan"),
                "p_left": float("nan"), "n_draws": 0, "draws": np.array([])}

    # Precompute the BHAR-at-every-entry vector for each kept ticker.
    pools: list[np.ndarray] = []
    for t in kept["ticker"]:
        s = prices[t].dropna()
        idx = s.index
        spy_al = spy.reindex(idx).ffill()
        p = s.to_numpy(dtype=float)
        m = spy_al.to_numpy(dtype=float)
        last = len(p) - window - 1
        if last <= 0:
            pools.append(np.array([np.nan]))
            continue
        p0, p1 = p[:last + 1], p[window: last + 1 + window]
        m0, m1 = m[:last + 1], m[window: last + 1 + window]
        with np.errstate(divide="ignore", invalid="ignore"):
            bh = (p1 / p0 - 1.0) - (m1 / m0 - 1.0)
        bh = bh[np.isfinite(bh)]
        pools.append(bh if bh.size else np.array([np.nan]))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for d in range(n_draws):
        vals = np.array([pool[rng.integers(0, pool.size)] for pool in pools])
        draws[d] = np.nanmean(vals)
    return {"obs": obs, "placebo_mean": float(np.nanmean(draws)),
            "placebo_sd": float(np.nanstd(draws, ddof=1)),
            "p_left": float(np.mean(draws <= obs)), "n_draws": n_draws, "draws": draws}


# --------------------------------------------------------------------------- #
# The tradable overlay — "short the cursed sponsor, long SPY"
# --------------------------------------------------------------------------- #
def curse_overlay(spy: pd.Series, prices: dict[str, pd.Series], deals: pd.DataFrame,
                  window: int = 252, cost_bps: float = 5.0,
                  borrow_bps_yr: float = 100.0) -> dict:
    """Short the sponsor / long SPY for ``window`` sessions after each deal.

    If the curse is real, this book earns +BHAR (= −sponsor + SPY). Each deal is one
    round trip on a 2×-NAV book (one short leg + one long leg): 2 sides × one-way cost ×
    NAV per leg on entry and exit (4 × cost_bps total), plus borrow on the short leg for
    the holding period. Long-horizon, so borrow is charged pro-rata to ``window``.
    """
    bhar, kept = stack_bhar(spy, prices, deals, window)
    n = bhar.size
    if n == 0:
        return {"n": 0, "gross_mean": float("nan"), "net_mean": float("nan"),
                "t_net": float("nan"), "win_rate": float("nan")}
    # curse book return per deal = short sponsor + long SPY = -sponsor_ret + spy_ret = -BHAR?
    # BHAR = sponsor_ret - spy_ret, so short-sponsor/long-SPY earns -(sponsor_ret) + spy_ret
    # = -BHAR - 2*spy? No: it earns (spy_ret) + (-sponsor_ret). Relative to cash it's
    # spy_ret - sponsor_ret = -BHAR. Market-neutral book: pnl = -BHAR.
    book = -bhar
    round_trip = 4.0 * cost_bps / 1e4              # 2 legs × entry+exit
    borrow = (borrow_bps_yr / 1e4) * (window / TRADING_DAYS)
    net = book - round_trip - borrow
    gross_mean, _ = one_sample_t(book)
    net_mean, t_net = one_sample_t(net)
    return {"n": n, "gross_mean": gross_mean, "net_mean": net_mean, "t_net": t_net,
            "win_rate": float((net > 0).mean()),
            "cost_drag_bps": (round_trip + borrow) * 1e4}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(spy: pd.Series, prices: dict[str, pd.Series],
                     events: list[tuple[str, pd.Timestamp]], window: int = 252) -> dict:
    """Run the headline cross-event BHAR test on a synthetic world."""
    deals = pd.DataFrame({"date": [d for _, d in events],
                          "ticker": [t for t, _ in events],
                          "venue": [t for t, _ in events]})
    cs = car_stats(spy, prices, deals, window)
    return {"n": cs["n"], "mean": cs["mean"], "t": cs["t"]}
