"""Real-tape verification — Study 1001 (The Leaky Fold). Regenerates docs/results.md.

Builds momentum and volatility features and forward-return labels on the real
tape, scores an identical ridge model under five validation schemes, decomposes the difference
between shuffled cross-validation and walk-forward into temporal leakage and label overlap,
sweeps the label horizon to show how the overlap leak scales, and checks on synthetic data with
a planted signal that purging removes the illusion without removing the information.

    python studies/1001-purged-cv-embargo/examples/verify.py            # cache-only
    python studies/1001-purged-cv-embargo/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from leakyfold import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


HORIZON = 20
K = 5
EMBARGO = 0.01
RIDGE = 1e-3


def report() -> dict:
    px_all = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "horizon": HORIZON, "k": K, "embargo": EMBARGO,
               "asset": data.EQUITY, "fingerprint": data.fingerprint(px_all)}

    px = px_all[data.EQUITY].dropna()
    F = st.make_features(px)
    L = st.make_labels(px, HORIZON)
    common = F.index.intersection(L.index)
    h["n_obs"] = int(len(common))
    h["n_features"] = int(F.shape[1])
    h["overlap_pct"] = st.overlap_fraction(HORIZON, 1)
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px_all)}")
    print(f"  {data.EQUITY}: {len(px):,} sessions -> {len(common):,} usable observations")
    print(f"  {F.shape[1]} features: {', '.join(F.columns)}")
    print(f"  label: the next {HORIZON} sessions' return")
    print(f"  two adjacent observations share {h['overlap_pct']:.0%} of their label window — "
          f"that overlap is the whole problem")

    print("\n=== 1. the same model, five validation schemes ===")
    c = st.compare_schemes(F, L, K, EMBARGO, RIDGE)
    print(c.round(4).to_string())
    h["schemes"] = c.reset_index().to_dict("records")
    h["shuffled_ic"] = float(c.loc["k-fold, shuffled", "ic"])
    h["shuffled_r2"] = float(c.loc["k-fold, shuffled", "r2"])
    h["sequential_ic"] = float(c.loc["k-fold, sequential", "ic"])
    h["purged_ic"] = float(c.loc["purged k-fold", "ic"])
    h["embargo_ic"] = float(c.loc["purged + embargo", "ic"])
    h["honest_ic"] = float(c.loc["walk-forward", "ic"])
    print(f"  shuffled says the model has an IC of {h['shuffled_ic']:+.4f}")
    print(f"  walk-forward says {h['honest_ic']:+.4f}")
    print(f"  the difference, {h['shuffled_ic'] - h['honest_ic']:+.4f}, is the illusion")

    print("\n=== 2. where does the illusion come from? ===")
    d = st.leakage_decomposition(F, L, K, EMBARGO)
    h.update({k: d[k] for k in ("temporal_leak", "overlap_leak", "embargo_effect",
                                "total_illusion", "overlap_share")})
    print(f"  seeing the future at all (shuffled - sequential): "
          f"{d['temporal_leak']:+.4f}")
    print(f"  LABEL OVERLAP (sequential - purged):              "
          f"{d['overlap_leak']:+.4f}")
    print(f"  the embargo (purged - purged+embargo):            "
          f"{d['embargo_effect']:+.4f}")
    print(f"  residual (purged+embargo - walk-forward):         "
          f"{d['embargo_ic'] - d['walk_forward_ic']:+.4f}")
    print(f"  total:                                            "
          f"{d['total_illusion']:+.4f}")
    print(f"  -> label overlap is {d['overlap_share']:.0%} of the illusion, and it happens")
    print(f"     with every date in the right order. 'Don't shuffle time series' does not")
    print(f"     protect you from it.")

    print("\n=== 2b. an aside that nearly derailed this study: how to average folds ===")
    pool_rows = []
    for kk in range(6):
        w0 = st.synthetic_panel(n=4000, predictability=0.0, horizon=HORIZON, seed=1001 + kk)
        F0, L0 = w0["features"], w0["labels"]
        cc0 = F0.index.intersection(L0.index)
        X0 = F0.loc[cc0].to_numpy(dtype=float)
        y0 = L0.loc[cc0, "label"].to_numpy(dtype=float)
        s0 = st.score_scheme(X0, y0, st.purged_kfold(L0.loc[cc0], K, EMBARGO), RIDGE)
        pool_rows.append({"seed": 1001 + kk, "per_fold": s0["ic"], "pooled": s0["ic_pooled"]})
    h["pooling_artefact"] = pool_rows
    h["pooled_mean"] = float(np.mean([r["pooled"] for r in pool_rows]))
    h["perfold_mean"] = float(np.mean([r["per_fold"] for r in pool_rows]))
    print(f"  on six signal-free worlds, purged k-fold scored by averaging the PER-FOLD")
    print(f"  correlations: {h['perfold_mean']:+.4f} — correctly near zero")
    print(f"  the same folds, scored by POOLING every prediction and correlating once: "
          f"{h['pooled_mean']:+.4f}")
    print("  pooling mixes clouds with different centres, so it measures their arrangement")
    print("  rather than skill. The artefact is larger than the leak being studied, and it")
    print("  points the wrong way, which is worse.")

    print("\n=== 3. how much data does purging cost? ===")
    plain = st.kfold_sequential(len(common), K)
    purged = st.purged_kfold(L.loc[common], K, 0.0)
    emb = st.purged_kfold(L.loc[common], K, EMBARGO)
    n_plain = float(np.mean([len(t) for t, _ in plain]))
    n_purged = float(np.mean([len(t) for t, _ in purged]))
    n_emb = float(np.mean([len(t) for t, _ in emb]))
    h["purged_data_loss"] = float(1 - n_purged / n_plain)
    h["embargo_data_loss"] = float(1 - n_emb / n_plain)
    print(f"  average training size: {n_plain:,.0f} plain -> {n_purged:,.0f} purged "
          f"({h['purged_data_loss']:.1%} removed) -> {n_emb:,.0f} with embargo "
          f"({h['embargo_data_loss']:.1%})")
    print("  that cost is the honest reason people skip this, and it is worth knowing before")
    print("  deciding the label horizon")

    print("\n=== 4. the leak against the label horizon ===")
    sw = st.horizon_sweep(px, horizons=(1, 5, 20, 60, 120), k=K, embargo=EMBARGO)
    print(sw.round(4).to_string())
    h["horizon_sweep"] = sw.reset_index().to_dict("records")
    h["leak_at_short"] = float(sw.loc[1, "overlap_leak"]) if 1 in sw.index else np.nan
    h["longest_horizon"] = int(sw.index.max())
    h["leak_at_long"] = float(sw.loc[sw.index.max(), "overlap_leak"])
    print(f"  at a 1-day label there is no overlap and the leak is {h['leak_at_short']:+.4f}")
    print(f"  at a {h['longest_horizon']}-day label it is {h['leak_at_long']:+.4f}")
    print("  the label horizon is a design choice, and this is its hidden cost")

    print("\n=== 5. every asset ===")
    cross = []
    for tk in data.TICKERS:
        if tk == data.CASH or tk not in px_all.columns:
            continue
        s = px_all[tk].dropna()
        if len(s) < 1500:
            continue
        f2, l2 = st.make_features(s), st.make_labels(s, HORIZON)
        d2 = st.leakage_decomposition(f2, l2, K, EMBARGO)
        if not d2:
            continue
        cross.append({"asset": tk, **{k: d2[k] for k in
                                      ("shuffled_ic", "sequential_ic", "purged_ic",
                                       "embargo_ic", "walk_forward_ic", "total_illusion",
                                       "overlap_share")}})
        print(f"  {tk:6s} shuffled {d2['shuffled_ic']:+.4f} -> walk-forward "
              f"{d2['walk_forward_ic']:+.4f}  (illusion {d2['total_illusion']:+.4f}, "
              f"{d2['overlap_share']:.0%} from overlap)")
    h["cross_asset"] = cross

    print("\n=== 6. does the fix destroy real signal? ===")
    ctrl = []
    for pred in (0.0, 0.5, 1.0, 2.0, 3.0):
        w = st.synthetic_panel(n=6000, predictability=pred, horizon=HORIZON)
        c2 = st.compare_schemes(w["features"], w["labels"], K, EMBARGO, RIDGE)
        ctrl.append({"planted": pred,
                     "shuffled": float(c2.loc["k-fold, shuffled", "ic"]),
                     "purged_embargo": float(c2.loc["purged + embargo", "ic"]),
                     "walk_forward": float(c2.loc["walk-forward", "ic"])})
        print(f"  planted {pred:.1f}: shuffled {ctrl[-1]['shuffled']:+.4f}, "
              f"purged+embargo {ctrl[-1]['purged_embargo']:+.4f}, "
              f"walk-forward {ctrl[-1]['walk_forward']:+.4f}")
    h["planted_control"] = ctrl
    strong = [r for r in ctrl if r["planted"] >= 1.0]
    h["planted_ic"] = 1.0
    h["planted_recovered"] = float(strong[0]["purged_embargo"]) if strong else np.nan
    h["planted_detected"] = bool(h["planted_recovered"] > 0.05)
    print(f"  with a planted signal the purged scheme recovers "
          f"{h['planted_recovered']:+.4f} — "
          f"{'it still finds real information' if h['planted_detected'] else 'it does NOT, which would be a problem'}")
    zero = [r for r in ctrl if r["planted"] == 0.0][0]
    print(f"  with nothing planted, shuffled reports {zero['shuffled']:+.4f} and "
          f"purged+embargo reports {zero['purged_embargo']:+.4f}")

    print("\n=== 7. how many folds, how big an embargo? ===")
    grid = []
    for k in (3, 5, 10):
        for e in (0.0, 0.01, 0.05, 0.10):
            c3 = st.compare_schemes(F, L, k, e, RIDGE)
            grid.append({"k": k, "embargo": e,
                         "purged_embargo_ic": float(c3.loc["purged + embargo", "ic"]),
                         "walk_forward_ic": float(c3.loc["walk-forward", "ic"]),
                         "gap": float(c3.loc["purged + embargo", "ic"]
                                      - c3.loc["walk-forward", "ic"])})
            print(f"  k={k:2d}, embargo {e:.0%}: purged+embargo "
                  f"{grid[-1]['purged_embargo_ic']:+.4f} vs walk-forward "
                  f"{grid[-1]['walk_forward_ic']:+.4f} (gap {grid[-1]['gap']:+.4f})")
    h["grid"] = grid

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    schemes = "\n".join(
        f"| {r['scheme']} | {int(r['n']):,} | {int(r['mean_train_size']):,} | "
        f"{r['ic']:+.4f} | {r['r2']:+.4f} |" for r in h["schemes"])
    sw = "\n".join(
        f"| {int(r['horizon'])} | {r['overlap_with_neighbour']:.0%} | "
        f"{r['shuffled_ic']:+.4f} | {r['sequential_ic']:+.4f} | {r['purged_ic']:+.4f} | "
        f"{r['walk_forward_ic']:+.4f} | **{r['overlap_leak']:+.4f}** |"
        for r in h["horizon_sweep"])
    cross = "\n".join(
        f"| {r['asset']} | {r['shuffled_ic']:+.4f} | {r['sequential_ic']:+.4f} | "
        f"{r['purged_ic']:+.4f} | {r['walk_forward_ic']:+.4f} | "
        f"**{r['total_illusion']:+.4f}** | {r['overlap_share']:.0%} |"
        for r in h["cross_asset"])
    ctrl = "\n".join(
        f"| {r['planted']:.1f} | {r['shuffled']:+.4f} | {r['purged_embargo']:+.4f} | "
        f"{r['walk_forward']:+.4f} |" for r in h["planted_control"])
    grid = "\n".join(
        f"| {r['k']} | {r['embargo']:.0%} | {r['purged_embargo_ic']:+.4f} | "
        f"{r['walk_forward_ic']:+.4f} | {r['gap']:+.4f} |" for r in h["grid"])
    return f"""# Results — Study 1001 (The Leaky Fold) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_obs']:,} observations,
{h['n_features']} features, a {h['horizon']}-session forward label, {h['k']} folds. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The same model, five ways of validating it

Identical features, identical labels, identical ridge regression. **Only the validation scheme
differs.**

| Scheme | Scored | Mean training size | Information coefficient | R² |
|---|--:|--:|--:|--:|
{schemes}

Shuffled cross-validation says the model has an IC of **{h['shuffled_ic']:+.4f}**. Walk-forward
— the only scheme that resembles how a model is actually used — says **{h['honest_ic']:+.4f}**.

## 2. Where the illusion comes from

| Channel | Contribution |
|---|--:|
| Seeing the future at all (shuffled − sequential) | {h['temporal_leak']:+.4f} |
| **Label overlap (sequential − purged)** | **{h['overlap_leak']:+.4f}** |
| The embargo (purged − purged+embargo) | {h['embargo_effect']:+.4f} |
| Residual (purged+embargo − walk-forward) | {h['embargo_ic'] - h['honest_ic']:+.4f} |
| **Total illusion** | **{h['total_illusion']:+.4f}** |

Label overlap is **{h['overlap_share']:.0%}** of the illusion — and it occurs with **every date
in the correct order**. Two adjacent observations share {h['overlap_pct']:.0%} of their label
window, so a training point next to the test block is very nearly a test point. The standard
advice, "don't shuffle time series", does nothing about this.

## 2b. How the folds are averaged, which nearly derailed this study

Scoring a cross-validation scheme has an obvious implementation and a correct one, and they are
not the same. Pooling every out-of-fold prediction into one vector and correlating it with the
labels once measures the *arrangement of the per-fold prediction clouds* — each fold is
standardised on its own training set, so each sits at its own level — rather than the skill
inside them.

| Purged k-fold on six signal-free worlds | Reported IC |
|---|--:|
| Mean of the per-fold correlations (used throughout) | {h['perfold_mean']:+.4f} |
| Pooled: concatenate predictions, correlate once | **{h['pooled_mean']:+.4f}** |

The artefact is **larger than the leak this study measures**, and it points the wrong way, which
is worse than being large: it makes the honest schemes look actively harmful. It is pinned in
[`tests/test_strategy.py`](../tests/test_strategy.py) as
`test_pooling_predictions_across_folds_invents_a_negative_ic`.

## 3. What purging costs

Average training set: purging removes **{h['purged_data_loss']:.1%}** of it, and the embargo
takes it to {h['embargo_data_loss']:.1%}. That cost is the honest reason people skip the fix,
and it should inform the choice of label horizon before it informs the choice of validation
scheme.

## 4. The leak against the label horizon

| Horizon | Overlap with neighbour | Shuffled | Sequential | Purged | Walk-forward | Overlap leak |
|---|--:|--:|--:|--:|--:|--:|
{sw}

At a one-day label there is no overlap and nothing to purge. At {h['longest_horizon']} days each
label shares {st_overlap(h)} of its window with the next one, and the overlap leak grows from
{h['leak_at_short']:+.4f} to **{h['leak_at_long']:+.4f}**. The label horizon is a modelling
choice with a validation cost attached, and the two are usually decided separately.

## 5. Every asset

| Asset | Shuffled | Sequential | Purged | Walk-forward | Illusion | From overlap |
|---|--:|--:|--:|--:|--:|--:|
{cross}

## 6. Does the fix destroy real signal?

A fix that removed the information along with the leakage would be no fix. On synthetic data
with a **planted** relationship of known strength:

| Planted strength | Shuffled | Purged + embargo | Walk-forward |
|---|--:|--:|--:|
{ctrl}

Read the first row: with **nothing** planted, shuffled cross-validation still reports skill.
Read the rest: the purged estimate rises with the planted signal, so it detects real information
rather than merely reporting zero.

## 7. Folds and embargo size

| k | Embargo | Purged + embargo | Walk-forward | Gap |
|---|--:|--:|--:|--:|
{grid}

## Caveats

- **One model family.** A ridge regression on seven features. A model with more capacity would
  exploit leakage *harder*, so the illusion measured here is a lower bound for anything more
  flexible — which is the direction that matters for the conclusion.
- **Walk-forward is treated as the truth.** It is the most honest scheme available, not a
  perfect one: it uses less data, so its estimate is noisier and its own fitting is done on
  shorter histories. The residual gap in section 2 partly reflects that rather than remaining
  leakage.
- **No hyperparameter search.** Real pipelines tune on the validation folds too, which
  compounds the problem substantially — a point studies **996** and **860** cover.
- **Purging is implemented on label windows only.** Features built from long lookbacks also
  reach backward across fold boundaries; a fully rigorous implementation would purge on the
  union of the feature and label windows, which would remove more data still.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1001-purged-cv-embargo](../README.md). Not investment advice.*
"""


def st_overlap(h: dict) -> str:
    """Format the overlap share at the longest horizon swept."""
    return f"{st.overlap_fraction(int(h['longest_horizon']), 1):.1%}"

def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    print("##HEADLINE## " + json.dumps(h, default=float))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
