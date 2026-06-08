"""Debounce, the feed->events join + coverage, and the momentum control."""

import pandas as pd

from social_oracle import mentions


def _feed(rows):
    return pd.DataFrame(rows, columns=["timestamp", "ticker"]).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"])
    )


def test_first_mentions_debounces_within_cooldown():
    feed = _feed([
        ("2026-01-01", "AAA"),
        ("2026-01-03", "AAA"),   # within 5 days -> dropped
        ("2026-01-09", "AAA"),   # >5 days later -> kept
        ("2026-01-02", "BBB"),   # different name -> kept
    ])
    deb = mentions.first_mentions(feed, cooldown_days=5)
    got = list(zip(deb["ticker"], deb["timestamp"].dt.strftime("%Y-%m-%d")))
    assert ("AAA", "2026-01-01") in got and ("AAA", "2026-01-09") in got
    assert ("AAA", "2026-01-03") not in got
    assert ("BBB", "2026-01-02") in got


def test_to_events_coverage_accounts_for_every_mention(panel, feed):
    events, cov = mentions.to_events(feed, panel)
    # coverage is an honest funnel: debounce <= raw, events == len(events)
    assert cov["raw"] == len(feed)
    assert cov["after_debounce"] <= cov["raw"]
    assert cov["events"] == len(events)
    # the funnel adds up: debounced = events + dropped
    assert cov["after_cohort"] == cov["events"] + cov["no_price"] + cov["too_close_to_edge"]


def test_to_events_drops_names_without_price(panel, feed):
    poisoned = pd.concat([feed, _feed([("2026-01-15", "NOPE")])], ignore_index=True)
    _, cov = mentions.to_events(poisoned, panel)
    assert cov["no_price"] >= 1


def test_event_columns_and_window_bounds(panel, feed):
    events, _ = mentions.to_events(feed, panel, pre=5, horizon=21)
    assert list(events.columns) == ["ticker", "entry_pos", "entry_date"]
    for t, p in zip(events["ticker"], events["entry_pos"]):
        n = len(panel[t])
        assert p - 5 >= 0 and p + 21 < n   # full window, no look-ahead


def test_hot_streak_events_are_well_formed(panel):
    hot = mentions.hot_streak_events(panel)
    assert list(hot.columns) == ["ticker", "entry_pos", "entry_date"]
    assert len(hot) > 0
