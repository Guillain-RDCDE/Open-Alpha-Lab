"""The signals are pure functions of the panel and the conditions nest as expected."""

import pandas as pd

from crimson_hour import signals


def test_confluence_is_oc_red_and_ib_rejected(feat):
    conf = signals.confluence(feat)
    expect = signals.oc_red(feat) & signals.ib_high_rejected(feat)
    assert (conf.fillna(False) == expect.fillna(False)).all()


def test_masks_partition(feat):
    m = signals.condition_masks(feat)
    # oc_red and oc_green are complementary and cover everything
    assert (m["oc_red"].astype(bool) ^ m["oc_green"].astype(bool)).all()
    # the confluence is a subset of oc_red
    assert (m["confluence (oc_red & ib_rej)"].fillna(False) <= m["oc_red"]).all()
    # confluence and "oc_red & not-rejected" are disjoint and together make oc_red
    union = (m["confluence (oc_red & ib_rej)"].fillna(False)
             | m["oc_red & ib_NOT_rej"].fillna(False))
    assert (union == m["oc_red"]).all()


def test_baseline_is_all_true(feat):
    m = signals.condition_masks(feat)
    assert m["baseline"].all()
    assert len(m["baseline"]) == len(feat)


def test_na_ib_flag_propagates():
    """When IB-rejection is NA (hourly bars), the confluence is NA, not silently False."""
    f = pd.DataFrame({"oc_red": [True], "ib_high_rejected": [pd.NA],
                      "session_red": [True], "rest_red": [True]})
    assert pd.isna(signals.confluence(f).iloc[0])
