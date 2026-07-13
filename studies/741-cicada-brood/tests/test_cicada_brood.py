"""Fully offline, deterministic tests for Study 741 (Cicada-Brood).

No network: everything runs on the hardcoded brood table and the seeded synthetic world.
Asserts pure-function identities and the positive-control's two required properties (the
null does not systematically fire; a planted bump is recovered).

    pytest -q studies/741-cicada-brood/tests
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cicada_brood import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The brood calendar (hardcoded — offline)
# --------------------------------------------------------------------------- #
def test_brood_years_are_distinct_and_deduped():
    ys = data.brood_years()
    assert ys == sorted(set(ys))                 # distinct + sorted
    assert len(ys) == 24                         # 24 distinct emergence years
    assert 2024 in ys                            # the dual 17x13 co-emergence
    # 2024 has two brood rows but is one event year
    rows_2024 = data.brood_table().query("year == 2024")
    assert len(rows_2024) == 2 and 2024 in ys


def test_seventeen_year_subset_and_famous_subset():
    all_y = set(data.brood_years())
    y17 = set(data.brood_years(cycle=17))
    fam = set(data.brood_years(famous_only=True))
    assert y17.issubset(all_y) and fam.issubset(all_y)
    assert fam == {2004, 2007, 2013, 2021, 2024}  # the marquee national events
    assert len(y17) == 23


# --------------------------------------------------------------------------- #
# Inference primitives (pure)
# --------------------------------------------------------------------------- #
def test_one_sample_t_matches_closed_form():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    out = st.one_sample_t(x)
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert out["n"] == 5
    assert np.isclose(out["mean"], x.mean())
    assert np.isclose(out["t"], x.mean() / se)


def test_one_sample_t_on_zero_mean_is_small():
    # symmetric data -> mean ~ 0 -> |t| small
    x = np.array([-0.02, 0.02, -0.01, 0.01])
    assert abs(st.one_sample_t(x)["t"]) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(18, 24)
    assert lo < 18 / 24 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_welch_zero_when_groups_identical():
    a = np.array([0.01, 0.02, 0.03])
    assert np.isclose(st.welch_t(a, a.copy()), 0.0)


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, no network)
# --------------------------------------------------------------------------- #
def test_synthetic_world_is_deterministic():
    c1, e1 = data.synthetic_world(bump=0.0, seed=741)
    c2, e2 = data.synthetic_world(bump=0.0, seed=741)
    assert e1 == e2
    assert np.allclose(c1.to_numpy(), c2.to_numpy())


def test_null_world_does_not_systematically_fire():
    ts = np.array([st.synthetic_detect(*data.synthetic_world(bump=0.0, seed=741 + s),
                                       k=data.WINDOW_K)["t"] for s in range(20)])
    # an unbiased detector: mean t near 0 and at most a couple of ~5% false positives
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2).sum() <= 3


def test_planted_bump_is_recovered():
    close, em = data.synthetic_world(bump=0.04, seed=741)
    out = st.synthetic_detect(close, em, k=data.WINDOW_K)
    assert out["t"] > 3.0          # a clear, planted signal lights up
    assert out["mean"] > 0.02      # abnormal CAR is materially positive


# --------------------------------------------------------------------------- #
# Event table shape (on the synthetic tape — offline)
# --------------------------------------------------------------------------- #
def test_event_table_one_row_per_resolved_year_and_cost_identity():
    close, em = data.synthetic_world(bump=0.0, seed=741)
    ar = st.abnormal_returns(st.daily_returns(close))
    ev = st.build_event_table(close, ar, em, k=data.WINDOW_K, cost_bps=5.0)
    # one row per emergence year whose full window sits on the tape (the last year's
    # window can run off the edge and is dropped, never zero-filled)
    assert len(em) - 1 <= len(ev) <= len(em)
    assert set(ev["year"]).issubset(set(em))
    assert list(ev["year"]) == sorted(ev["year"])          # ordered, non-overlapping
    # net = gross - 2 * one-way cost (one round trip)
    assert np.allclose(ev["ret_net"], ev["ret_gross"] - 2 * 5.0 * 1e-4)
