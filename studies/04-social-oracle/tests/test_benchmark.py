"""Random-day null and the momentum control."""

import numpy as np

from social_oracle import benchmark, mentions


def test_car_forward_shape_and_tail_nan(panel):
    frame = next(iter(panel.values()))
    car = benchmark.car_forward(frame, 5)
    assert len(car) == len(frame)
    assert np.isnan(car[-5:]).all() and not np.isnan(car[0])


def test_conditional_vs_unconditional_columns_and_identity(panel, events):
    tbl = benchmark.conditional_vs_unconditional(panel, events, horizons=(1, 5), n_iter=200)
    for col in ("mean_cond", "mean_uncond", "excess", "p_greater"):
        assert col in tbl.columns
    assert np.allclose(tbl["excess"], tbl["mean_cond"] - tbl["mean_uncond"])


def test_pvalue_in_unit_interval(panel, events):
    tbl = benchmark.conditional_vs_unconditional(panel, events, horizons=(5,), n_iter=200)
    p = tbl["p_greater"].iloc[0]
    assert 0.0 <= p <= 1.0


def test_excess_vs_alternative_gap_identity(panel, events):
    hot = mentions.hot_streak_events(panel)
    tbl = benchmark.excess_vs_alternative(panel, events, hot, horizons=(5,), n_iter=200)
    if not tbl.empty:
        row = tbl.iloc[0]
        assert np.isclose(row["gap"], row["mean_mention"] - row["mean_alt"])
        assert 0.0 <= row["p_mention_gt_alt"] <= 1.0


def test_fade_is_detected_as_negative_excess(panel, events):
    # The baked-in fade means the mention's forward abnormal return undershoots a
    # random day at a month: excess should be <= 0.
    tbl = benchmark.conditional_vs_unconditional(panel, events, horizons=(21,), n_iter=200)
    assert tbl["excess"].iloc[0] <= 0.0
