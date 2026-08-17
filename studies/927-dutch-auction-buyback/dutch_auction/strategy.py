"""Event study + inference for Study 927 — Dutch Auction self-tender offers.

A *modified Dutch auction* self-tender is the loudest signal a board can send: instead of
dribbling out open-market repurchases, the company posts a price range, invites holders to
name their price, and buys a large block in a single, fixed, twenty-business-day window.
The folklore says it marks the bottom — management, who know the most, are paying *above*
the market for their own stock, and the shrunken share count does the rest.

This module measures four separate things and refuses to blur them:

1. **The announcement move** ``ar0`` — the event-day abnormal return (stock − SPY, day
   ``t``). This is the *news*, not an edge: the offer becomes public that day, so nobody
   trading on the filing can capture it.
2. **The run-up** ``pre5`` — the (−5, −1) abnormal return. Some issuers pre-announce via an
   SC TO-C or a press release; this quantifies how much of the pop was already leaking.
3. **The tender window** ``window`` — enter at the close of ``t+1`` (**one** execution lag:
   the filing is public at the close of day ``t``, acted on at ``t+1``), hold ``EXPIRY_TD``
   = 20 sessions to ``t+21`` — the first session after Rule 14e-1's twenty-business-day
   statutory minimum — measured excess of SPY. This *is* tradable.
4. **The post-expiry drift** ``m1`` / ``m3`` / ``m6`` — abnormal return from the expiry
   proxy over the next 21 / 63 / 126 trading days. The "does it mark the bottom?" claim.

**PROXY / ASSUMPTION.** Real expiry dates are not in the EDGAR full-text index, so expiry
is proxied by ``t + 20`` trading days — the statutory minimum an issuer tender must stay
open (Rule 14e-1). Offers that are extended or repriced expire later; ``expiry_sweep``
re-runs the whole post-expiry analysis at 15 / 20 / 25 / 30 trading days.

Inference. Events overlap in calendar time, so a naive one-sample *t* across events
overstates precision. We therefore report three things side by side: the cross-event
one-sample *t*, a **leave-one-out jackknife** of that *t*, and a **calendar-time
portfolio** (equal-weight every name currently inside its window, daily, HAC/Newey-West
*t*) which is the standard fix for event clustering. A **placebo** re-draws the same names
on random dates, and a **circular block bootstrap** gives the mean a CI.

Costs. The tradable leg is long the issuer and short SPY (that is what "excess of SPY"
means when you actually deploy it). Costs are charged **one-way × the position's actual
portfolio weight**, on the day it is opened and again on the day it is closed. The weight
matters: the sleeve is equal-weight across whichever names are live, so an event that
opens alongside two others is only a third of NAV and can only cost a third of a
round-trip. Charging a full-NAV round trip per event — the obvious shortcut — overstates
the drag by the average number of live names (1.55× on this sample) and would let the
cost model, rather than the tape, carry a negative verdict. The SPY short pays **borrow**
on the sleeve's gross short notional, swept because index-ETF borrow is cheap but not
free. *Not* modelled: the intra-sleeve rebalancing turnover when a name enters or leaves
and the survivors re-weight — second-order next to the open/close round trips, and named
here rather than silently omitted. Small-cap issuers are the ones you cannot trade at
10 bps — hence the liquidity screen in ``screen_events``.

**A NAMED BIAS THAT LIVES IN THIS FILE.** ``_screen`` requires ``ENTRY_LAG + EXPIRY_TD +
126 + 1`` = 148 sessions of tape *after* the event, so any issuer that was acquired, taken
private or delisted within roughly seven months of its own tender is dropped. That is
survivorship, and its direction is knowable: a self-tender is sometimes the opening move
of a going-private transaction, and a takeout is exactly the outcome that would print a
large positive six-month abnormal return. The screen therefore removes events that would
push the post-expiry drift *up*, which flatters this study's own "there is no drift"
conclusion. The drift result should be read as an upper bound on how empty it is, not a
neutral measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Window geometry, in trading days relative to the event day t (= index position 0).
PRE_START, PRE_END = -6, -1        # the (−5, −1) run-up, measured close(-6) → close(-1)
ENTRY_LAG = 1                      # the ONE execution lag: act at the close of t+1
EXPIRY_TD = 20                     # PROXY: Rule 14e-1 minimum, 20 business days
HORIZONS = {"m1": 21, "m3": 63, "m6": 126}


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
class EventPanel:
    """Aligned issuer/market arrays plus the screened event positions.

    Holding the panel as plain numpy arrays (stock close, market close sampled on the
    stock's own trading days) makes the placebo — hundreds of thousands of window
    returns — cheap, and keeps every window definition in one place.
    """

    def __init__(self, prices: dict, events, market: str = "SPY",
                 min_price: float = 5.0, min_pre_days: int = 252,
                 dollar_volume: dict = None, min_dollar_volume: float = 0.0):
        self.market = market
        self.min_price = float(min_price)
        self.min_pre_days = int(min_pre_days)
        self.min_dollar_volume = float(min_dollar_volume)
        if market not in prices:
            raise KeyError(f"market leg {market!r} missing from prices")
        mkt = prices[market]
        self.arrays = {}
        for tk, s in prices.items():
            if tk == market:
                continue
            pos = mkt.index.searchsorted(s.index)
            pos = np.clip(pos, 0, len(mkt) - 1)
            self.arrays[tk] = (s.index, s.to_numpy(dtype=float),
                               mkt.to_numpy(dtype=float)[pos])
        self.dollar_volume = dollar_volume or {}
        self.events = self._screen(events)

    # -- screens ---------------------------------------------------------- #
    def _screen(self, events):
        """Apply the objective, pre-registered screens and return the kept events.

        Every screen is mechanical and stated up front: a cached tape for the issuer, at
        least ``min_pre_days`` sessions before the event, enough sessions after it to close
        the longest post-expiry window, a $5 minimum price (a penny-stock screen), and —
        when the optional dollar-volume cache is present — a median 60-session dollar
        volume floor. No event is dropped by inspecting its outcome.
        """
        need_after = ENTRY_LAG + EXPIRY_TD + max(HORIZONS.values()) + 1
        kept = []
        for row in events:
            tk, date = row[0], row[1]
            a = self.arrays.get(tk)
            if a is None:
                continue
            idx, px, _ = a
            p = int(idx.searchsorted(pd.Timestamp(date)))
            if p < self.min_pre_days or p + need_after >= len(px):
                continue
            if px[p] < self.min_price:
                continue
            if self.min_dollar_volume > 0.0:
                dv = self.dollar_volume.get(tk)
                if dv is None:
                    continue
                q = dv.index.searchsorted(idx[p])
                if q < 60:
                    continue
                if float(np.median(dv.to_numpy(dtype=float)[q - 60:q])) < self.min_dollar_volume:
                    continue
            kept.append({"ticker": tk, "date": str(date), "pos": p, "t0": idx[p],
                         "accession": row[2] if len(row) > 2 else "",
                         "name": row[3] if len(row) > 3 else tk,
                         "price": float(px[p])})
        return kept

    # -- window arithmetic ------------------------------------------------ #
    def abn(self, ticker: str, a: int, b: int) -> float:
        """Abnormal (stock − market) simple return between index positions ``a`` and ``b``."""
        _, px, mk = self.arrays[ticker]
        return float((px[b] / px[a] - 1.0) - (mk[b] / mk[a] - 1.0))

    def abn_vec(self, ticker: str, a, b) -> np.ndarray:
        _, px, mk = self.arrays[ticker]
        a = np.asarray(a); b = np.asarray(b)
        return (px[b] / px[a] - 1.0) - (mk[b] / mk[a] - 1.0)

    def n_obs(self, ticker: str) -> int:
        return len(self.arrays[ticker][1])


# --------------------------------------------------------------------------- #
# The event table
# --------------------------------------------------------------------------- #
def event_table(panel: EventPanel, expiry_td: int = EXPIRY_TD,
                shift: int = 0) -> pd.DataFrame:
    """Abnormal returns for every screened event, one row per event.

    ``shift`` moves the assumed event day by that many trading days — the date-robustness
    sweep, which answers "does the result survive if the announcement date is off by a
    few days?" without any re-fitting.
    """
    rows = []
    for e in panel.events:
        tk = e["ticker"]
        p = e["pos"] + int(shift)
        n = panel.n_obs(tk)
        last = p + ENTRY_LAG + expiry_td + max(HORIZONS.values())
        if p + PRE_START < 0 or last >= n:
            continue
        x = p + ENTRY_LAG + expiry_td      # the expiry proxy position
        r = {
            "ticker": tk, "name": e["name"], "t0": e["t0"], "date": e["date"],
            "accession": e["accession"], "price": e["price"],
            "ar0": panel.abn(tk, p - 1, p),
            "pre5": panel.abn(tk, p + PRE_START, p + PRE_END),
            "window": panel.abn(tk, p + ENTRY_LAG, x),
        }
        for lab, h in HORIZONS.items():
            r[lab] = panel.abn(tk, x, x + h)
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def auto_nw_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    return newey_west_t(x, lags=int(np.floor(4.0 * (len(x) / 100.0) ** (2.0 / 9.0))))


def jackknife_t(x) -> dict:
    """Leave-one-out range of the one-sample *t* — is the result one lucky event?"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return {"t": float("nan"), "t_min": float("nan"), "t_max": float("nan")}
    ts = np.array([one_sample_t(np.delete(x, i)) for i in range(len(x))])
    ts = ts[np.isfinite(ts)]        # a leave-one-out sample can be degenerate
    if ts.size == 0:
        return {"t": one_sample_t(x), "t_min": float("nan"), "t_max": float("nan"),
                "n": int(len(x))}
    return {"t": one_sample_t(x), "t_min": float(ts.min()), "t_max": float(ts.max()),
            "n": int(len(x))}


def bootstrap_mean_ci(x, n_boot: int = 5000, seed: int = 927,
                      alpha: float = 0.05, block: int = 1) -> dict:
    """Percentile CI for the mean. ``block`` > 1 gives a circular block bootstrap.

    Events are ordered by date, so a block bootstrap on that ordering preserves the
    clustering of events in calendar time (2015 and 2021 were busy years) that an i.i.d.
    resample would destroy.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    if block <= 1:
        draws = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    else:
        nb = int(np.ceil(n / block))
        offs = np.arange(block)
        starts = rng.integers(0, n, size=(n_boot, nb))
        idx = (starts[:, :, None] + offs[None, None, :]).reshape(n_boot, -1) % n
        draws = x[idx[:, :n]].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(n), "frac_negative": float((draws < 0).mean())}


# --------------------------------------------------------------------------- #
# Placebo — the same names, random dates
# --------------------------------------------------------------------------- #
def placebo_null(panel: EventPanel, metrics=("ar0", "window", "m6"),
                 n_draws: int = 2000, seed: int = 927,
                 expiry_td: int = EXPIRY_TD) -> pd.DataFrame:
    """Re-draw each event's date uniformly at random on the *same* issuer's own tape.

    Holds the sample of names, their volatility and their listing histories fixed and
    destroys only the *timing*. The share of draws whose mean is at least as extreme as
    the observed one is the placebo *p* — the honest null for an event study whose names
    were selected by a filing search rather than sampled at random.
    """
    rng = np.random.default_rng(seed)
    need_after = ENTRY_LAG + expiry_td + max(HORIZONS.values()) + 1
    tickers, los, his = [], [], []
    for e in panel.events:
        tk = e["ticker"]
        n = panel.n_obs(tk)
        lo, hi = panel.min_pre_days, n - need_after
        if hi <= lo:
            continue
        tickers.append(tk); los.append(lo); his.append(hi)
    los = np.asarray(los); his = np.asarray(his)
    out = {m: np.empty(n_draws) for m in metrics}
    for b in range(n_draws):
        p = rng.integers(los, his)
        acc = {m: np.empty(len(tickers)) for m in metrics}
        for i, tk in enumerate(tickers):
            pi = int(p[i])
            x = pi + ENTRY_LAG + expiry_td
            if "ar0" in acc:
                acc["ar0"][i] = panel.abn(tk, pi - 1, pi)
            if "pre5" in acc:
                acc["pre5"][i] = panel.abn(tk, pi + PRE_START, pi + PRE_END)
            if "window" in acc:
                acc["window"][i] = panel.abn(tk, pi + ENTRY_LAG, x)
            for lab, h in HORIZONS.items():
                if lab in acc:
                    acc[lab][i] = panel.abn(tk, x, x + h)
        for m in metrics:
            out[m][b] = acc[m].mean()
    return pd.DataFrame(out)


def placebo_p(observed: float, draws: np.ndarray) -> float:
    """Two-sided placebo *p*: share of random-date means at least as extreme."""
    d = np.asarray(draws, dtype=float)
    return float((np.abs(d) >= abs(observed)).mean())


# --------------------------------------------------------------------------- #
# Calendar-time portfolio — the fix for overlapping events
# --------------------------------------------------------------------------- #
def calendar_time_portfolio(panel: EventPanel, tbl: pd.DataFrame,
                            hold_days: int = EXPIRY_TD, start_offset: int = ENTRY_LAG,
                            cost_bps: float = 10.0, borrow_bps_yr: float = 30.0,
                            cash: pd.Series = None) -> dict:
    """Equal-weight, daily-rebalanced portfolio of every name currently inside its window.

    Each event contributes ``stock − SPY`` daily returns from ``t + start_offset`` for
    ``hold_days`` sessions; the portfolio is the equal-weight average across whichever
    events are live that day (flat, i.e. in cash, on days with none). Because it is a
    genuine daily time series, its **HAC/Newey-West** *t* is immune to the overlapping-event
    problem that inflates the cross-event *t*.

    Costs: ``cost_bps`` one-way on **both** legs when a slot is opened and again when it is
    closed, charged **at the position's actual weight** (``1 / n_live`` on that day, because
    the sleeve is equal-weight across live names) and booked on the day it happens rather
    than smeared. A full-NAV round trip per event would overstate the drag by the average
    number of live names. Plus ``borrow_bps_yr`` on the short-SPY notional every day the
    sleeve is on. Returns gross and net series, their annualised Sharpes and HAC *t*s.
    """
    mkt_index = None
    daily, legs = {}, []
    for e in panel.events:
        tk = e["ticker"]
        idx, px, mk = panel.arrays[tk]
        if mkt_index is None:
            mkt_index = idx
        p = e["pos"]
        a, b = p + start_offset, p + start_offset + hold_days
        if b >= len(px):
            continue
        r = (px[a + 1:b + 1] / px[a:b] - 1.0) - (mk[a + 1:b + 1] / mk[a:b] - 1.0)
        dts = idx[a + 1:b + 1]
        for d, v in zip(dts, r):
            daily.setdefault(d, []).append(v)
        legs.append((dts[0], dts[-1]))          # the day it is opened, the day it is closed

    if not daily:
        return {}
    days = sorted(daily)
    gross = pd.Series([float(np.mean(daily[d])) for d in days],
                      index=pd.DatetimeIndex(days, name="date"))
    # Union calendar: flat (0.0) on days with no live event, so the Sharpe is the Sharpe
    # of the sleeve as actually deployed, not of its busy days only.
    full = pd.date_range(gross.index[0], gross.index[-1], freq="B")
    gross = gross.reindex(full).fillna(0.0)
    n_live = pd.Series([len(daily.get(d, [])) for d in full], index=full)

    n_events = len(legs)
    live_days = int((n_live > 0).sum())
    # Weight-aware, day-stamped costs: 2 legs × one-way × the slot's own weight, twice.
    cost_day = pd.Series(0.0, index=full)
    turnover = 0.0
    for d_in, d_out in legs:
        for d in (d_in, d_out):
            w = 1.0 / max(len(daily.get(d, [])), 1)
            cost_day.loc[d] = cost_day.get(d, 0.0) + 2.0 * cost_bps * 1e-4 * w
            turnover += 2.0 * w                 # two legs, at weight, per open and per close
    borrow = borrow_bps_yr * 1e-4 / TRADING_DAYS
    net = gross - cost_day - (n_live > 0).astype(float) * borrow

    def stats(s, cash_leg=None):
        # The sleeve is dollar-neutral (long issuer, short SPY), so the cash collateral
        # earns exactly the financing rate and its return is ALREADY excess-of-cash —
        # no BIL subtraction is needed here. The long-only cash race lives in
        # ``long_only_race``, where the cash leg genuinely matters.
        r = s.astype(float)
        sd = r.std(ddof=1)
        return {
            "mean_daily_bps": float(r.mean() * 1e4),
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
            "vol_ann": float(sd * np.sqrt(TRADING_DAYS)),
            "t_hac": auto_nw_t(r.to_numpy()),
            "n_days": int(len(r)),
            "cum": float((1.0 + r).prod() - 1.0),
        }

    return {
        "gross": stats(gross), "net": stats(net),
        "n_events": n_events, "live_days": live_days,
        "avg_names_live": float(n_live[n_live > 0].mean()),
        "nav_round_trips": float(turnover / 4.0),   # in full-NAV-round-trip units
        "cost_drag_bps_day": float(cost_day.mean() * 1e4),
        "series_gross": gross, "series_net": net,
    }


def long_only_race(panel: EventPanel, market_prices: pd.Series, cash_prices: pd.Series,
                   hold_days: int = EXPIRY_TD, start_offset: int = ENTRY_LAG,
                   cost_bps: float = 10.0) -> dict:
    """Race the **long-only** event basket against SPY, both **excess of cash** (BIL).

    On days with a live event the sleeve is an equal-weight long basket of those issuers;
    on days without one it sits in cash. Both arms are then measured minus BIL's actual
    total return, so the Sharpe comparison is excess-of-cash vs excess-of-cash. Costs are
    ``cost_bps`` one-way charged **at the slot's own weight** (``1 / n_live``) on the day it
    opens and the day it closes — one leg only, a long-only basket has no short and pays no
    borrow. The SPY arm is charged a single round trip for the one buy and the one sell that
    a buy-and-hold benchmark actually needs; over 4,000 sessions that is worth well under
    a thousandth of a basis point a day, so it is booked for symmetry rather than for effect.
    """
    daily, legs = {}, []
    for e in panel.events:
        tk = e["ticker"]
        idx, px, _ = panel.arrays[tk]
        p = e["pos"]
        a, b = p + start_offset, p + start_offset + hold_days
        if b >= len(px):
            continue
        r = px[a + 1:b + 1] / px[a:b] - 1.0
        dts = idx[a + 1:b + 1]
        for d, v in zip(dts, r):
            daily.setdefault(d, []).append(v)
        legs.append((dts[0], dts[-1]))
    if not daily:
        return {}

    cash_r = cash_prices.pct_change()
    days = cash_r.index[(cash_r.index >= min(daily)) & (cash_r.index <= max(daily))]
    c = cash_r.reindex(days).fillna(0.0).to_numpy(dtype=float)
    m = market_prices.reindex(days).ffill().pct_change().fillna(0.0).to_numpy(dtype=float)
    live = np.array([len(daily.get(d, [])) for d in days])
    basket = np.array([float(np.mean(daily[d])) if d in daily else 0.0 for d in days])
    basket = np.where(live > 0, basket, c)          # parked in cash when no event is live

    n_events = len(legs)
    live_days = int((live > 0).sum())
    pos = {d: i for i, d in enumerate(days)}
    cost_day = np.zeros(len(days))
    for d_in, d_out in legs:
        for d in (d_in, d_out):
            i = pos.get(d)
            if i is not None:
                cost_day[i] += cost_bps * 1e-4 / max(len(daily.get(d, [])), 1)
    net = basket - cost_day
    # Symmetry: the benchmark arm buys once and sells once, at the same one-way cost.
    m = m.copy()
    m[0] -= cost_bps * 1e-4
    m[-1] -= cost_bps * 1e-4

    def stats(r):
        e = r - c
        sd = e.std(ddof=1)
        return {"sharpe_excess_cash": float(e.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
                "mean_daily_bps": float(e.mean() * 1e4),
                "vol_ann": float(sd * np.sqrt(TRADING_DAYS)),
                "t_hac": auto_nw_t(e)}

    diff = (net - c) - (m - c)
    return {
        "basket_gross": stats(basket), "basket_net": stats(net), "spy": stats(m),
        "sharpe_adv_net": stats(net)["sharpe_excess_cash"] - stats(m)["sharpe_excess_cash"],
        "t_diff_hac": auto_nw_t(diff),
        "n_days": int(len(days)), "live_days": live_days, "n_events": n_events,
    }


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def cost_sweep(panel: EventPanel, tbl: pd.DataFrame,
               cost_grid=(0.0, 5.0, 10.0, 25.0, 50.0),
               borrow_bps_yr: float = 30.0, cash: pd.Series = None) -> list:
    """Net window return and calendar-time Sharpe across one-way cost assumptions."""
    out = []
    for c in cost_grid:
        cal = calendar_time_portfolio(panel, tbl, cost_bps=c,
                                      borrow_bps_yr=borrow_bps_yr, cash=cash)
        out.append({
            "cost_bps": c,
            "net_window_pct": float(tbl["window"].mean() - 4.0 * c * 1e-4) * 100,
            "sharpe_net": cal["net"]["sharpe"], "t_hac_net": cal["net"]["t_hac"],
            "mean_daily_bps": cal["net"]["mean_daily_bps"],
        })
    return out


def borrow_sweep(panel: EventPanel, tbl: pd.DataFrame,
                 borrow_grid=(0.0, 30.0, 100.0, 300.0),
                 cost_bps: float = 10.0, cash: pd.Series = None) -> list:
    """The short-SPY borrow assumption swept — it is an ASSUMPTION, not a tape input."""
    out = []
    for b in borrow_grid:
        cal = calendar_time_portfolio(panel, tbl, cost_bps=cost_bps,
                                      borrow_bps_yr=b, cash=cash)
        out.append({"borrow_bps_yr": b, "sharpe_net": cal["net"]["sharpe"],
                    "t_hac_net": cal["net"]["t_hac"],
                    "mean_daily_bps": cal["net"]["mean_daily_bps"]})
    return out


def expiry_sweep(panel: EventPanel, grid=(15, 20, 25, 30)) -> list:
    """Sweep the expiry PROXY. The post-expiry drift must not hinge on the 20-day guess."""
    out = []
    for e in grid:
        t = event_table(panel, expiry_td=e)
        row = {"expiry_td": e, "n": int(len(t)),
               "window_pct": float(t["window"].mean()) * 100,
               "t_window": one_sample_t(t["window"])}
        for lab in HORIZONS:
            row[f"{lab}_pct"] = float(t[lab].mean()) * 100
            row[f"t_{lab}"] = one_sample_t(t[lab])
        out.append(row)
    return out


def date_shift_sweep(panel: EventPanel, grid=(-5, -2, -1, 0, 1, 2, 5)) -> list:
    """Sweep the assumed event day. A date-locked effect must peak at shift 0."""
    out = []
    for s in grid:
        t = event_table(panel, shift=s)
        out.append({"shift": s, "n": int(len(t)),
                    "ar0_pct": float(t["ar0"].mean()) * 100,
                    "t_ar0": one_sample_t(t["ar0"])})
    return out


def era_cut(tbl: pd.DataFrame, split: str = "2018-01-01",
            metrics=("ar0", "window", "m6")) -> dict:
    """Split the events at ``split`` and re-run the cross-event tests on each half."""
    out = {}
    for tag, m in [("early", tbl["t0"] < pd.Timestamp(split)),
                   ("late", tbl["t0"] >= pd.Timestamp(split))]:
        d = tbl[m]
        row = {"n": int(len(d))}
        for c in metrics:
            row[f"{c}_pct"] = float(d[c].mean()) * 100
            row[f"t_{c}"] = one_sample_t(d[c])
            row[f"win_{c}"] = float((d[c] > 0).mean())
        out[tag] = row
    return out


def summarise(tbl: pd.DataFrame,
              metrics=("pre5", "ar0", "window", "m1", "m3", "m6")) -> pd.DataFrame:
    """Mean / median / *t* / hit-rate for every measured window, one row each."""
    rows = []
    for c in metrics:
        x = tbl[c].dropna()
        rows.append({"metric": c, "n": int(len(x)),
                     "mean_pct": float(x.mean()) * 100,
                     "median_pct": float(x.median()) * 100,
                     "sd_pct": float(x.std(ddof=1)) * 100,
                     "t": one_sample_t(x),
                     "hit_rate": float((x > 0).mean())})
    return pd.DataFrame(rows).set_index("metric")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel_prices: dict, events, market: str = "MKT") -> dict:
    """Run the event study on a synthetic panel with a known planted effect.

    On the planted world the announcement-day mean and the 6-month drift must both be
    recovered close to the planted sizes with a large *t*; on the null both must sit near
    zero. Proves the harness is unbiased — it never supports a real-tape stamp.
    """
    panel = EventPanel(panel_prices, events, market=market, min_price=0.0,
                       min_pre_days=252)
    tbl = event_table(panel)
    return {
        "n": int(len(tbl)),
        "ar0_bps": float(tbl["ar0"].mean()) * 1e4,
        "t_ar0": one_sample_t(tbl["ar0"]),
        "m6_bps": float(tbl["m6"].mean()) * 1e4,
        "t_m6": one_sample_t(tbl["m6"]),
        "window_bps": float(tbl["window"].mean()) * 1e4,
        "t_window": one_sample_t(tbl["window"]),
    }
