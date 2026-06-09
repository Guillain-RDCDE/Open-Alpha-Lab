"""The beta-neutral BAB alpha is real and HAC-significant on the anomaly tape and a statistical zero on
the null; the low-vol book is honestly sub-1 beta; the inference is the discriminator the raw Sharpe
sort isn't."""

import numpy as np

from dull_roar import decompose


def test_bab_alpha_significant_with_anomaly(anomaly_panel, anomaly_market):
    bab = decompose.beta_neutral_bab(anomaly_panel, market=anomaly_market, cost_bps=1.0)
    assert bab["alpha_ann_pct"] > 3.0          # a real, sizeable beta-neutral alpha
    assert bab["alpha_t"] > 3.0                # HAC-significant
    assert abs(bab["beta"]) < 0.1              # genuinely beta-neutral by construction
    assert bab["beta_low_leg"] < bab["beta_high_leg"]


def test_bab_alpha_is_zero_on_null(null_panel, null_market):
    bab = decompose.beta_neutral_bab(null_panel, market=null_market, cost_bps=1.0)
    assert abs(bab["alpha_t"]) < 2.0           # no significant alpha when alpha is baked to zero


def test_low_vol_book_is_sub_one_beta(anomaly_panel, anomaly_market):
    reg = decompose.capm_alpha(
        decompose.build(anomaly_panel, market=anomaly_market)["low_only"], anomaly_market)
    assert 0.0 < reg["beta"] < 1.0             # the low-vol book runs less market risk -> part of its edge


def test_beta_tilt_test_prices_the_beta_story(anomaly_panel, anomaly_market):
    tilt = decompose.beta_tilt_test(anomaly_panel, market=anomaly_market, cost_bps=1.0)
    assert tilt["low_beta"] < 1.0
    assert tilt["sharpe_gain"] > 0.0           # low-vol does lift Sharpe on the anomaly tape
    # the lift is partly just lower beta: the at-beta-1 excess is a *different* (smaller-context) number
    assert np.isfinite(tilt["excess_cagr_at_beta1_pct"])


def test_null_kills_everything(null_panel, null_market):
    tilt = decompose.beta_tilt_test(null_panel, market=null_market, cost_bps=1.0)
    boot = decompose.sharpe_gain_bootstrap(null_panel, market=null_market, n_boot=500, cost_bps=1.0)
    assert abs(tilt["alpha_ann_pct"]) < 2.0
    assert boot["ci_low"] < 0.0 < boot["ci_high"]   # the long-only edge CI straddles zero on the null
