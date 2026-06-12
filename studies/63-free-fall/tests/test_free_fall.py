"""The synthetic world is deterministic; the short-vol carry has a positive mean day but catastrophic
negative skew and a deep worst day when crashes are present; the null has no crash tail."""
import numpy as np
from free_fall import data, strategy as st


def test_world_deterministic(crash_world):
    df, truth = crash_world
    df2, _ = data.synthetic_shortvol(crash_prob=0.003, seed=63)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_crash_risk


def test_carry_is_positive_on_normal_days(crash_world):
    df, _ = crash_world
    assert st.carry_vs_crash(df["SVXY"])["median_day_bp"] > 0.0   # steady carry on quiet days


def test_crash_world_has_fat_left_tail(crash_world):
    df, _ = crash_world
    s = st.summary(df["SVXY"])
    assert s["skew"] < -1.0                 # the steamroller — strong negative skew
    assert s["worst_day"] < -0.2            # a catastrophic single day


def test_null_world_no_crash(null_world):
    df, _ = null_world
    s = st.summary(df["SVXY"])
    assert s["worst_day"] > -0.2            # no catastrophic days
    assert s["skew"] > -1.0


def test_worst_day_returns_date(crash_world):
    df, _ = crash_world
    wd, wdt = st.worst_day(df["SVXY"])
    assert wd == df["SVXY"].min()
