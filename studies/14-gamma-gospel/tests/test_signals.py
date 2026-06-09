"""The regime labels partition the panel and read the right column."""

from gamma_gospel import signals


def test_regimes_partition_the_panel(panel):
    neg = signals.neg_gamma(panel)
    pos = signals.pos_gamma(panel)
    assert (neg ^ pos).all()                      # exactly one of the two holds every day
    assert int(neg.sum()) + int(pos.sum()) == len(panel)


def test_masks_present(panel):
    m = signals.regime_masks(panel)
    assert set(m) == {"baseline", "neg_gamma", "pos_gamma"}
    assert m["baseline"].all()


def test_outcomes_read_their_columns(panel):
    assert signals.realized_vol(panel).equals(panel["rv"].astype(float))
    assert signals.directional_efficiency(panel).equals(panel["de"].astype(float))
