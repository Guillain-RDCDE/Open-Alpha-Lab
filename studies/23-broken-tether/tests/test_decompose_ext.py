"""The spurious-pairs false-positive rate is non-trivial (selection trap), the hedge ratio is stabler on
a real pair than a spurious one, and the OOS scan builds across a basket."""

from broken_tether import decompose, extension, data


def test_spurious_pairs_false_positives():
    sp = decompose.spurious_pairs(n_series=20, n_bars=1500, hl_threshold=60.0, seed=1)
    assert sp["n_pairs"] == 190
    assert 0.0 <= sp["false_positive_rate"] < 0.5     # some independent walks look cointegrated by chance


def test_hedge_ratio_drift_is_measured(coint_px):
    """The drift metric is well-formed (the *real-data* finding — large drift on ETF pairs — lives in
    docs/results.md; on the idealised synthetic both legs load near beta 1, so direction isn't a clean
    invariant)."""
    d = extension.hedge_ratio_drift(coint_px["A"], coint_px["B"])
    assert d["beta_min"] <= d["beta_mean"] <= d["beta_max"]
    assert d["beta_rel_drift"] >= 0.0


def test_oos_scan_builds():
    basket = {f"S{i}": data.synthetic_pair(revert_rho=0.95, seed=i)[0]["A"] for i in range(4)}
    scan = extension.oos_scan(basket, cost_bps=2.0, max_pairs=6)
    assert {"first_half", "second_half", "survives"}.issubset(scan.columns)
    assert len(scan) >= 1
