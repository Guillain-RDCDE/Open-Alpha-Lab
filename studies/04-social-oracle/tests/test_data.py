"""The feed loader, the panel builder, and the synthetic universe."""

import numpy as np
import pandas as pd

from social_oracle import data


def test_load_feed_normalizes_columns(tmp_path):
    csv = tmp_path / "feed.csv"
    csv.write_text(
        "created_at,cashtag,sentiment\n"
        "2026-01-02, $sive ,80\n"
        "2026-01-03,AXTI,55\n",
        encoding="utf-8",
    )
    feed = data.load_feed(str(csv))
    assert list(feed.columns) == ["timestamp", "ticker", "score"]
    assert list(feed["ticker"]) == ["SIVE", "AXTI"]   # upper-cased, $ stripped, trimmed
    assert pd.api.types.is_datetime64_any_dtype(feed["timestamp"])


def test_build_panel_attaches_returns_and_market(panel):
    for frame in panel.values():
        assert {"r_cc", "r_mkt"}.issubset(frame.columns)
        assert not frame["r_mkt"].isna().any()   # market filled, never NaN


def test_synthetic_panel_shape(universe):
    panel, feed = universe
    assert len(panel) == 20
    assert len(feed) == 120
    assert {"timestamp", "ticker", "score"}.issubset(feed.columns)


def test_equal_weight_market_is_average(panel):
    # r_mkt on a common date equals the mean of the names' r_cc there (no benchmark passed)
    a_frame = next(iter(panel.values()))
    d = a_frame.index[100]
    rcc = [f["r_cc"].get(d, np.nan) for f in panel.values()]
    assert np.isclose(a_frame["r_mkt"].loc[d], np.nanmean(rcc), atol=1e-9)
