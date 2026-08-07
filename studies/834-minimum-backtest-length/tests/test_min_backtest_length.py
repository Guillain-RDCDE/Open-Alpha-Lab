"""Offline, fixed-seed tests for the Minimum-Backtest-Length machinery.

The MinTRL formula matches its (Z/SR)^2 rule of thumb; the requirement grows as the Sharpe
falls and as the tail gets fatter/more left-skewed; MinTRL is the length at which PSR = conf;
the synthetic generator hits its planted Sharpe and moments; a worthless world's PSR test is
calibrated at its nominal 5% and its short backtests still post gaudy Sharpes by luck; a genuine
edge is only reliably confirmed past its power-length. All deterministic and offline.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from min_backtest_length import data, strategy as st  # noqa: E402
from scipy import stats  # noqa: E402

Z95 = float(stats.norm.ppf(0.95))
FREQ = data.TRADING_DAYS


# ---- the MinTRL formula ----------------------------------------------------
def test_min_trl_matches_rule_of_thumb():
    """Gaussian daily MinTRL is within ~2% of the (Z/SR)^2 rule of thumb."""
    for sr in (0.25, 0.5, 1.0, 2.0):
        exact = st.min_trl_years(sr, freq=FREQ, conf=0.95)
        rule = (Z95 / sr) ** 2
        assert abs(exact - rule) / rule < 0.02


def test_min_trl_grows_as_sharpe_falls():
    sr_grid = [2.0, 1.0, 0.5, 0.25]
    mtl = st.min_trl_curve(sr_grid, freq=FREQ, conf=0.95)
    assert np.all(np.diff(mtl) > 0)                 # strictly increasing as SR falls
    assert mtl[3] > 40 and mtl[1] < 3               # Sharpe-0.25 > 40yr; Sharpe-1 < 3yr


def test_min_trl_infinite_when_no_edge():
    assert st.min_trl_years(0.0, freq=FREQ) == float("inf")
    assert st.min_trl_years(-0.5, freq=FREQ) == float("inf")


def test_psr_equals_conf_at_min_trl():
    """By construction PSR at n = MinTRL equals the confidence level."""
    for sr in (0.5, 1.0, 2.0):
        n = st.min_trl_years(sr, freq=FREQ, conf=0.95)
        psr = st.probabilistic_sharpe_ratio(sr, n, freq=FREQ, sr_star_ann=0.0)
        assert abs(psr - 0.95) < 1e-6


def test_negative_skew_and_fat_tails_lengthen_requirement():
    """A negatively-skewed, leptokurtic tape needs a LONGER track (monthly, where it bites)."""
    g = st.min_trl_years(1.0, freq=data.MONTHS, conf=0.95, skew=0.0, kurt=3.0)
    sk = st.min_trl_years(1.0, freq=data.MONTHS, conf=0.95, skew=-2.0, kurt=9.0)
    assert sk > g * 1.3          # fat left tail inflates MinTRL by >30% at monthly SR=1


def test_power_length_exceeds_min_trl():
    """The 95%-power length is materially longer than MinTRL (MinTRL is only ~50% power)."""
    mtl = st.min_trl_years(1.0, freq=FREQ, conf=0.95)
    powl = st.min_trl_for_power(1.0, freq=FREQ, conf=0.95, power=0.95)
    assert powl > mtl * 3


# ---- the synthetic generator ----------------------------------------------
def test_generator_hits_target_sharpe(skilled_returns, null_returns):
    ret1, _ = skilled_returns
    ret0, _ = null_returns
    assert abs(st.sharpe_ratio(ret1, FREQ) - 1.0) < 0.15   # planted Sharpe ~1 over a long tape
    assert abs(st.sharpe_ratio(ret0, FREQ)) < 0.15         # null ~0


def test_generator_moments_match_closed_form(fat_left_returns):
    ret, truth = fat_left_returns
    skew, kurt = st.sample_moments(ret)
    assert truth.skew < -1.5 and truth.kurt > 7          # planted shape (skew -2, kurt 9)
    assert abs(skew - truth.skew) < 0.3                   # sample skew tracks the truth
    assert kurt > 6.0                                     # heavy tails present


def test_deterministic():
    a, _ = data.synthetic_panel(sr_ann=0.0, n_sims=100, n_years=2.0, seed=834)
    b, _ = data.synthetic_panel(sr_ann=0.0, n_sims=100, n_years=2.0, seed=834)
    assert np.allclose(a, b)


# ---- the pitfall: short backtests can't tell skill from luck ----------------
def test_short_backtest_luck():
    lp = st.luck_prob(data, threshold_sr=1.0, n_years=2.0, freq=FREQ, n_sims=4000, seed=834)
    assert lp["frac"] > 0.04           # a worthless strategy posts Sharpe>=1 well above 5% at 2yr
    assert lp["best_sr"] > 2.0         # the luckiest worthless backtest looks spectacular


def test_null_psr_is_calibrated():
    """The correction is unbiased: on the null the PSR test fires at ~its nominal 5%."""
    s = st.simulate(data, sr_ann_true=0.0, n_years=2.0, freq=FREQ, n_sims=4000, conf=0.95, seed=834)
    assert 0.03 < s["reject_frac"] < 0.08     # nominal 5%, Monte-Carlo band around it
    assert s["reject_lo"] < 0.05 < s["reject_hi"]


# ---- the positive control: genuine edge IS detectable, past the length ------
def test_positive_control_power_rises_and_confirms():
    pc = st.power_curve(data, sr_ann_true=1.0, year_grid=(1.0, 2.71, 10.82),
                        freq=FREQ, n_sims=4000, conf=0.95, seed=834)
    fr = pc["reject_frac"]
    assert fr[0] < fr[1] < fr[2]              # detection rate climbs with the track length
    assert 0.40 < fr[1] < 0.60               # ~50% at MinTRL (observed-equals-target)
    assert fr[2] > 0.90                       # ~95% only at the power-length


# ---- inference primitives ---------------------------------------------------
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(50, 1000)
    assert lo < 0.05 < hi


def test_sharpe_ratio_degenerate():
    assert np.isnan(st.sharpe_ratio(np.zeros(100)))
    assert np.isnan(st.sharpe_ratio(np.array([0.01, 0.02])))
