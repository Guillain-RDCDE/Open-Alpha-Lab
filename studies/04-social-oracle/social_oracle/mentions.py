"""From a raw mention feed to clean, de-duplicated *events* — and the controls we
test them against.

A mention is not yet an event. Before it can go into an event study it has to be:

    * **joined to a price** — the name has to exist in the panel (a delisted or
      never-listed cashtag is dropped, and *counted*, not silently skipped);
    * **anchored to a session** — we set ``t=0`` at the first trading session on or
      after the mention's timestamp (the mention *day*), so the event study reads
      the information content of the call itself. The realistic *entry* — you can't
      buy the close you already missed — is the **next open**, charged in the
      backtest, not here;
    * **debounced** — a name she tweets ten times in a week is one *episode*, not
      ten independent observations. :func:`first_mentions` collapses a cluster to
      its first call, exactly as Study 02/03's ``first_crossings`` debounces a
      level that stays triggered.

The output container is a plain ``DataFrame`` with columns ``ticker, entry_pos,
entry_date`` — the same shape every downstream module (eventstudy, benchmark,
backtest, robustness) consumes.

Two *controls* also live here, because the whole question is "does the mention add
anything?":

    * :func:`hot_streak_events` — the **momentum** control. Days a name was already
      running (top-quantile trailing return). If the mention doesn't beat *this*,
      the influencer is just narrating momentum the tape already had.
    * the **coverage** report returned by :func:`to_events` — every mention we
      dropped, and why. Survivorship and selection are the headline risks of a feed
      like this; this is how the study states them in the open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COOLDOWN_DAYS = 5   # calendar days: one hype episode per name per ~week
DEFAULT_PRE = 5             # sessions of pre-event path we require (the run-up leg)
DEFAULT_HORIZON = 21        # sessions of forward path we require (≈ one month)


def first_mentions(feed: pd.DataFrame, cooldown_days: int = DEFAULT_COOLDOWN_DAYS) -> pd.DataFrame:
    """Collapse repeat mentions of the same ticker into fresh *episodes*.

    Within ``cooldown_days`` (calendar) of a kept mention of a given ticker, further
    mentions of that ticker are dropped. The rising-edge analogue of
    ``triggers.first_crossings``: a name spammed daily counts once, so the event
    study isn't dominated by whichever stock the feed happens to repeat most.
    """
    if feed.empty:
        return feed.copy()
    kept = []
    cooldown = pd.Timedelta(days=cooldown_days)
    last: dict[str, pd.Timestamp] = {}
    for row in feed.sort_values("timestamp").itertuples(index=False):
        t, ts = row.ticker, row.timestamp
        if t in last and ts - last[t] < cooldown:
            continue
        last[t] = ts
        kept.append(row)
    return pd.DataFrame(kept, columns=feed.columns).reset_index(drop=True)


def _entry_pos(frame_index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """Position of the first session on or after ``ts`` (-1 if none)."""
    pos = int(frame_index.searchsorted(pd.Timestamp(ts.date()), side="left"))
    return pos if pos < len(frame_index) else -1


def to_events(
    feed: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    *,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
    pre: int = DEFAULT_PRE,
    horizon: int = DEFAULT_HORIZON,
    min_mentions: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Resolve a feed into clean events, and report exactly what was dropped.

    Steps, in order: debounce (:func:`first_mentions`), optional ``min_mentions``
    cohort filter (keep only names mentioned at least that often — the "favourites"),
    join to the panel, anchor ``t=0`` to the mention session, and require a full
    ``[-pre, +horizon]`` window so no path is partial and there's no look-ahead.

    Returns ``(events, coverage)`` where ``events`` has columns
    ``ticker, entry_pos, entry_date`` and ``coverage`` is a dict of counts:
    ``raw, after_debounce, after_cohort, no_price, too_close_to_edge, events`` — so
    a study can print, e.g., "of 437 mentions, 74 had no tradeable price and 38 sat
    too close to the sample edge; 211 events remain." No silent caps.
    """
    cov = {"raw": int(len(feed))}
    deb = first_mentions(feed, cooldown_days=cooldown_days)
    cov["after_debounce"] = int(len(deb))

    if min_mentions > 1:
        counts = feed["ticker"].value_counts()
        keep = set(counts[counts >= min_mentions].index)
        deb = deb[deb["ticker"].isin(keep)]
    cov["after_cohort"] = int(len(deb))

    no_price = too_close = 0
    rows = []
    for row in deb.itertuples(index=False):
        t = row.ticker
        frame = panel.get(t)
        if frame is None or frame.empty:
            no_price += 1
            continue
        p = _entry_pos(frame.index, row.timestamp)
        if p < 0 or p - pre < 0 or p + horizon >= len(frame):
            too_close += 1
            continue
        rows.append({"ticker": t, "entry_pos": p, "entry_date": frame.index[p]})

    cov["no_price"] = int(no_price)
    cov["too_close_to_edge"] = int(too_close)
    events = pd.DataFrame(rows, columns=["ticker", "entry_pos", "entry_date"])
    cov["events"] = int(len(events))
    return events, cov


def frequent_names(feed: pd.DataFrame, k: int = 10) -> list[str]:
    """The ``k`` most-mentioned tickers — the influencer's 'favourites' cohort."""
    return list(feed["ticker"].value_counts().head(k).index)


def hot_streak_events(
    panel: dict[str, pd.DataFrame],
    *,
    lookback: int = 5,
    quantile: float = 0.90,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
    pre: int = DEFAULT_PRE,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """The **momentum control**: days a name was already on a tear.

    For every name, flag sessions whose trailing ``lookback``-day return sits in the
    top ``quantile`` of *that name's* history, debounce them by ``cooldown_days``,
    and require the same full window as :func:`to_events`. Feeding these into
    :func:`social_oracle.benchmark.excess_vs_alternative` answers the question that
    decides the study: *does being mentioned beat simply being hot?* If not, the
    "oracle" is a momentum sensor with a follower count.
    """
    cooldown = max(1, cooldown_days)  # in sessions here (control is panel-native)
    rows = []
    for t, frame in panel.items():
        close = frame["Close"].to_numpy()
        n = len(close)
        if n <= lookback + horizon:
            continue
        trail = np.full(n, np.nan)
        trail[lookback:] = close[lookback:] / close[:-lookback] - 1.0
        thr = np.nanquantile(trail, quantile)
        hot = trail >= thr
        last = -10**9
        for p in np.flatnonzero(hot):
            if p - last < cooldown or p - pre < 0 or p + horizon >= n:
                continue
            last = p
            rows.append({"ticker": t, "entry_pos": int(p), "entry_date": frame.index[p]})
    return pd.DataFrame(rows, columns=["ticker", "entry_pos", "entry_date"])
