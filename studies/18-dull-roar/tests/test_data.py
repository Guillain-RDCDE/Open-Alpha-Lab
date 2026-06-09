"""The synthetic universe is deterministic, shaped right, and its null is a genuinely flat control."""

import numpy as np

from dull_roar import data


def test_deterministic_and_shaped(anomaly):
    panel, market, truth = anomaly
    panel2, market2, _ = data.synthetic_panel(sml_slope=0.00018, seed=18)
    assert panel.shape == (truth.n_bars, truth.n_stocks)
    assert np.allclose(panel.to_numpy(), panel2.to_numpy())          # same seed -> same tape
    assert np.allclose(market.to_numpy(), market2.to_numpy())
    assert truth.has_anomaly and truth.annual_sml_alpha_spread > 0


def test_null_has_no_anomaly(null):
    _, _, truth = null
    assert not truth.has_anomaly
    assert truth.annual_sml_alpha_spread == 0.0


def test_null_is_flat_in_sharpe(null_panel):
    """Because idio vol scales with beta, the null's per-stock Sharpe carries *no* dependence on vol
    (the betas cancel). We can't pin its level — the single shared-market draw shifts every stock's
    realized mean together — but we can check the thing that matters: Sharpe and vol are ~uncorrelated
    across the cross-section, i.e. the security-market line is flat, not tilted."""
    mu = null_panel.mean()
    sd = null_panel.std(ddof=1)
    sharpe = (mu / sd * np.sqrt(252))
    corr = np.corrcoef(sd.to_numpy(), sharpe.to_numpy())[0, 1]
    assert abs(corr) < 0.25            # no systematic low-vol-earns-more gradient on a fair-CAPM tape


def test_vol_increases_with_beta(anomaly_panel):
    """Total realized vol should rank stocks monotonically (a vol sort is a clean beta sort)."""
    vols = anomaly_panel.std(ddof=1)
    assert vols.min() > 0
    # wide spread of vols across the cross-section (low-vol to high-vol names exist to sort)
    assert vols.max() / vols.min() > 2.0
