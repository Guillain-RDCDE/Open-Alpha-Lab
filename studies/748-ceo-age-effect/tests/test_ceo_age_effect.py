"""Fully offline, deterministic tests for Study 748 (CEO-Age-Effect).

No network: every test runs on the curated table or the deterministic synthetic panel.
Run: pytest -q studies/748-ceo-age-effect/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ceo_age_effect import data, strategy as st


def test_curated_ages_bucketing_offline():
    ages = data.curated_ages(score_date="2024-12-31", young_max=55)
    assert len(ages) == 40
    # Zuckerberg (1984) is young; Buffett (1930) is old; the cutoff is at age 55.
    assert ages.loc["META", "bucket"] == "young"
    assert ages.loc["BRK-B", "bucket"] == "old"
    assert (ages["age"] == 2024 - ages["birth_year"]).all()
    assert set(ages["bucket"]) == {"young", "old"}


def test_hac_mean_t_zero_mean_noise():
    # White noise has a mean t near zero; a big constant shift makes it large.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200) * 0.05
    assert abs(st.hac_mean_t(x)["t"]) < 2.5
    shifted = x + 0.10
    assert st.hac_mean_t(shifted)["t"] > 2.0


def test_capm_recovers_planted_beta():
    # ls = 1.4*mkt + noise  ->  beta ~ 1.4, alpha ~ 0.
    rng = np.random.default_rng(1)
    mkt = rng.standard_normal(120) * 0.04
    ls = 1.4 * mkt + rng.standard_normal(120) * 0.01
    ca = st.capm_alpha(ls, mkt)
    assert abs(ca["beta"] - 1.4) < 0.15
    assert abs(ca["alpha"]) < 0.005          # intercept economically ~zero
    assert ca["r2"] > 0.9                     # market explains almost all of ls


def test_synthetic_null_has_no_alpha_but_positive_beta():
    # At the null (age_alpha=0) young firms still carry a higher beta, but there is NO alpha.
    panel, mkt, truth = data.synthetic_panel(age_alpha=0.0, seed=748)
    assert not truth.has_effect
    frame = st.synthetic_ls(panel, mkt)
    ca = st.capm_alpha(frame["ls"].to_numpy(), frame["mkt"].to_numpy())
    assert abs(ca["alpha_t"]) < 2.0          # no false alpha
    assert ca["beta"] > 0.2                   # young-minus-old carries a real beta tilt


def test_synthetic_control_detects_planted_alpha_seed_robust():
    # House rule: average over >= 20 seeds. Null stays flat; a real premium clears t = 2.
    null = st.synthetic_mean_alpha_t(data, age_alpha=0.0, n_seeds=25)
    planted = st.synthetic_mean_alpha_t(data, age_alpha=0.008, n_seeds=25)
    assert abs(null["mean_alpha_t"]) < 1.0
    assert planted["mean_alpha_t"] > 2.0


def test_net_of_costs_reduces_return():
    panel, mkt, _ = data.synthetic_panel(age_alpha=0.006, seed=748)
    frame = st.synthetic_ls(panel, mkt)
    nc = st.net_of_costs(frame, cost_bps=5.0, borrow_ann_bps=75.0, annual_turnover=0.30)
    assert nc["net_ann"] < nc["gross_ann"]
    assert nc["cost_ann"] > 0
