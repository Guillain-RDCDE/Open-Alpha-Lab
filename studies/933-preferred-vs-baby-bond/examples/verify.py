"""Real-tape verification — Study 933 (Same Issuer, Two Ladders). Regenerates docs/results.md.

Reads the cached daily total-return closes of eight issuers that list **both** a preferred
and a baby bond, plus PFF (the sector benchmark) and BIL (the cash leg); races the two
ladders excess-of-cash, prints the pairwise same-issuer spread with its HAC *t*, the
block-bootstrap CIs, the per-issuer table, the era cut, the cost and borrow sweeps, the
stress episodes, the long-history two-pair extension, the redeemed-pair survivorship probe,
the alternative-rung cross-check and the PFF fallback. Network only on ``--fetch``.

    python studies/933-preferred-vs-baby-bond/examples/verify.py            # cache-only
    python studies/933-preferred-vs-baby-bond/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from two_ladders import data, strategy as st  # noqa: E402

COST_BPS = 25.0
SPLIT = "2024-04-01"

EPISODES = {
    "2022 rate shock":  ("2022-02-02", "2022-10-31"),
    "Mar-2023 banks":   ("2023-03-01", "2023-05-31"),
    "2024 issuer stress": ("2024-02-01", "2024-09-30"),
    "2025 tariff shock": ("2025-02-01", "2025-05-31"),
}

LONG_PAIRS = (
    ("CMS Energy",  "CMS-PB", "CMSC", "utility"),
    ("Duke Energy", "DUK-PA", "DUKB", "utility"),
)

ALT_PAIRS = (
    ("CMS Energy",           "CMS-PB", "CMSA",  "utility"),
    ("Duke Energy",          "DUK-PA", "DUKB",  "utility"),
    ("Brookfield Renewable", "BEP-PA", "BEPI",  "infrastructure"),
    ("Brookfield Infra.",    "BIP-PA", "BIPI",  "infrastructure"),
    ("B. Riley Financial",   "RILYP",  "RILYZ", "financial"),
    ("Oxford Lane Capital",  "OXLCP",  "OXLCZ", "closed-end credit"),
    ("Eagle Point Credit",   "ECC-PD", "ECCC",  "closed-end credit"),
    ("Babcock & Wilcox",     "BW-PA",  "BWNB",  "industrial"),
)


def _prep(px: pd.DataFrame, pairs, start: str):
    pref, bond = data.panel_returns(px, pairs=pairs, start=start)
    cash = px[data.CASH].pct_change().reindex(pref.index).dropna()
    return pref.loc[cash.index], bond.loc[cash.index], cash


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    pref, bond, cash = _prep(px, data.PAIRS, data.PANEL_START)

    legs = list(data.PREF_TICKERS + data.BOND_TICKERS)
    print(f"panel: {len(data.PAIRS)} issuer pairs, {pref.index[0].date()} -> {pref.index[-1].date()}  "
          f"n={len(pref)}  fp={data.fingerprint(px[legs].loc[pref.index])}")
    print(f"as-of {data.AS_OF}   cost {COST_BPS:.0f} bps one-way (PROXY)")

    cmp = st.compare(pref, bond, cash, cost_bps=COST_BPS)

    def fmt(s):
        return (f"exSharpe={s['sharpe']:+.3f}  CAGR={s['cagr']:+.2%}  Vol={s['vol_ann']:.2%}  "
                f"HAC t={s['tstat']:+.2f}")

    print(f"\nPreferred ladder : {fmt(cmp['pref'])}  MaxDD(abs)={cmp['dd_pref_abs']:+.2%}")
    print(f"Baby-bond ladder : {fmt(cmp['bond'])}  MaxDD(abs)={cmp['dd_bond_abs']:+.2%}")
    print(f"\nExcess-Sharpe advantage (pref - bond): {cmp['excess_sharpe_adv']:+.3f}")
    print(f"  ladder return spread : {cmp['ret_spread_ann']:+.2%}/yr  (HAC t = {cmp['t_ladder_diff']:+.2f})")
    print(f"  PAIRWISE same-issuer : {cmp['pairwise_spread_ann']:+.2%}/yr  (HAC t = {cmp['t_pairwise']:+.2f})")
    print(f"  vol ratio pref/bond  : {cmp['vol_ratio']:.2f}   annual turnover {cmp['turnover_ann_pref']:.0%}")

    print("\n=== bootstrap CIs (2000 draws, 21-day blocks, joint resampling) ===")
    for tag, s in [("pref ladder", cmp["e_pref"]), ("bond ladder", cmp["e_bond"])]:
        ci = st.bootstrap_ci(s, "sharpe", seed=933)
        print(f"  {tag:12s} exSharpe {ci['point']:+.3f}  95% CI [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]"
              f"  share<0 {ci['frac_negative']:.1%}")
    ci = st.bootstrap_sharpe_adv_ci(cmp["e_pref"], cmp["e_bond"], seed=933)
    print(f"  Sharpe advantage {ci['point']:+.3f}  95% CI [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]"
          f"  share<0 {ci['frac_negative']:.1%}")
    ci = st.bootstrap_ci(cmp["pairwise"], "mean", seed=933)
    print(f"  pairwise spread {ci['point']:+.2%}/yr  95% CI [{ci['ci_low']:+.2%}, {ci['ci_high']:+.2%}]")

    print("\n=== per-issuer table ===")
    tbl = st.per_pair_table(pref, bond)
    for iss, r in tbl.iterrows():
        print(f"  {iss:22s} pref CAGR {r['pref_cagr']:+7.2%} vol {r['pref_vol']:6.1%} DD {r['pref_dd']:+7.1%} | "
              f"bond CAGR {r['bond_cagr']:+7.2%} vol {r['bond_vol']:6.1%} DD {r['bond_dd']:+7.1%} | "
              f"spread {r['spread_ann']:+7.2%} (t {r['spread_t']:+5.2f})")
    hr = st.hit_rate(pref, bond)
    print(f"  wins (pref > bond): {hr['wins']}/{hr['n']}  Wilson 95% [{hr['ci_low']:.2f}, {hr['ci_high']:.2f}]")
    print(f"  median pair spread : {tbl['spread_ann'].median():+.2%}/yr")
    print(f"  pref vol > bond vol in {int((tbl['pref_vol'] > tbl['bond_vol']).sum())}/{len(tbl)} pairs")

    drop = "B. Riley Financial"
    pw_ex = st.pairwise_spread(pref.drop(columns=[drop]), bond.drop(columns=[drop]))
    print(f"  drop {drop}: pairwise {pw_ex.mean() * 252:+.2%}/yr (t = {st.hac_t(pw_ex):+.2f})")

    print("\n=== microstructure - how thin are these tapes? ===")
    legs = pd.concat(
        [pref.add_suffix(" [pref]"), bond.add_suffix(" [bond]")], axis=1
    )
    stale = st.staleness_table(legs)
    for leg, r in stale.iterrows():
        print(f"  {leg:32s} zero-days {r['zero_share']:5.1%}  AC1 {r['ac1']:+.2f}  vol {r['vol_ann']:6.1%}")
    print(f"  mean zero-day share: pref {stale.loc[stale.index.str.endswith('[pref]'), 'zero_share'].mean():.1%}"
          f"  bond {stale.loc[stale.index.str.endswith('[bond]'), 'zero_share'].mean():.1%}")

    print("\n=== is the RISK step a fact or a daily-tape artefact? ===")
    freq = st.vol_ratio_by_frequency(pref, bond)
    for f, r in freq.iterrows():
        print(f"  {f:8s} n={r['n_obs']:5.0f}  vol pref {r['vol_pref']:6.2%}  bond {r['vol_bond']:6.2%}"
              f"  ratio {r['vol_ratio']:.2f}   pref riskier in {r['pref_riskier']:.0f}/{r['n_pairs']:.0f} pairs")

    print("\n=== concentration probe - drop the two distressed issuers ===")
    for names in ([drop], [drop, "Babcock & Wilcox"]):
        d = st.drop_issuers(pref, bond, cash, names, cost_bps=COST_BPS)
        print(f"  ex {', '.join(names):40s} (n={d['n_pairs']}): pairwise {d['pairwise_spread_ann']:+.2%}/yr"
              f" (t={d['t_pairwise']:+.2f})  adv {d['excess_sharpe_adv']:+.3f}"
              f"  vol ratio {d['vol_ratio']:.2f} (monthly {d['vol_ratio_monthly']:.2f},"
              f" pref riskier {d['pref_riskier_monthly']}/{d['n_pairs']})"
              f"  DD {d['dd_pref_abs']:+.1%} vs {d['dd_bond_abs']:+.1%}")

    print(f"\n=== era cut (split {SPLIT}) ===")
    for tag, e in st.era_cut(pref, bond, cash, split=SPLIT, cost_bps=COST_BPS).items():
        if e is None:
            continue
        print(f"  {tag:5s} (n={e['n_days']:4d}): pref exSharpe {e['sharpe_pref']:+.2f} / bond {e['sharpe_bond']:+.2f}"
              f"  adv {e['excess_sharpe_adv']:+.2f}  pairwise {e['pairwise_spread_ann']:+.2%} (t={e['t_pairwise']:+.2f})"
              f"  DD {e['dd_pref_abs']:+.1%} vs {e['dd_bond_abs']:+.1%}")

    print("\n=== cost sweep (one-way PROXY) ===")
    for r in st.cost_sweep(pref, bond, cash):
        print(f"  cost={r['cost_bps']:5.1f}bps  pref {r['sharpe_pref']:+.3f} / bond {r['sharpe_bond']:+.3f}"
              f"  adv {r['excess_sharpe_adv']:+.3f}  spread {r['ret_spread_ann']:+.2%} (t={r['t_ladder_diff']:+.2f})")

    print("\n=== borrow sweep - long preferreds / short baby bonds (ASSUMPTION) ===")
    for r in st.borrow_sweep(pref, bond, cash, cost_bps=COST_BPS):
        print(f"  borrow={r['borrow_bps_ann']:5.0f}bp/yr  ret {r['ann_return']:+.2%}  Sharpe {r['sharpe']:+.3f}"
              f"  vol {r['vol_ann']:.1%}  DD {r['max_drawdown']:+.1%}  t {r['t']:+.2f}")

    print("\n=== stress episodes (cumulative, equal-weight legs) ===")
    stress = st.stress_table(pref, bond, EPISODES)
    for ep, r in stress.iterrows():
        print(f"  {ep:20s} ({r['days']:3.0f}d): pref {r['pref_pct']:+6.2f}%  bond {r['bond_pct']:+6.2f}%"
              f"  gap {r['gap_pp']:+6.2f} pp")

    print("\n=== long-history extension (CMS + Duke only, from 2019-04) ===")
    lp, lb, lc = _prep(px, LONG_PAIRS, "2019-04-01")
    lcmp = st.compare(lp, lb, lc, cost_bps=COST_BPS)
    print(f"  {lp.index[0].date()} -> {lp.index[-1].date()}  n={len(lp)}")
    print(f"  pref exSharpe {lcmp['pref']['sharpe']:+.3f} / bond {lcmp['bond']['sharpe']:+.3f}"
          f"  adv {lcmp['excess_sharpe_adv']:+.3f}  pairwise {lcmp['pairwise_spread_ann']:+.2%}"
          f" (t={lcmp['t_pairwise']:+.2f})  vol ratio {lcmp['vol_ratio']:.2f}")
    cov = st.stress_table(lp, lb, {"COVID crash": ("2020-02-19", "2020-03-23")})
    for ep, r in cov.iterrows():
        print(f"  {ep}: pref {r['pref_pct']:+.2f}%  bond {r['bond_pct']:+.2f}%  gap {r['gap_pp']:+.2f} pp")

    print("\n=== alternative-rung cross-check (CMSA/BEPI/BIPI instead of CMSC/BEPH/BIPH) ===")
    ap, ab, ac = _prep(px, ALT_PAIRS, data.PANEL_START)
    acmp = st.compare(ap, ab, ac, cost_bps=COST_BPS)
    print(f"  pairwise {acmp['pairwise_spread_ann']:+.2%}/yr (t={acmp['t_pairwise']:+.2f})"
          f"  adv {acmp['excess_sharpe_adv']:+.3f}  vol ratio {acmp['vol_ratio']:.2f}")

    print("\n=== survivorship probe - a pair the live screen throws away ===")
    dead = px[["SACH-PA", "SACC"]].dropna()
    dead = dead[dead.index >= pd.Timestamp("2021-11-15")]
    dr = dead.pct_change().dropna()
    dd = (dr["SACH-PA"] - dr["SACC"]).dropna()
    print(f"  Sachem Capital SACH-PA vs SACC (bond matured 2024-12): "
          f"{dead.index[0].date()} -> {dead.index[-1].date()}  n={len(dr)}")
    print(f"  pref cum {((1 + dr['SACH-PA']).prod() - 1) * 100:+.1f}%   bond cum {((1 + dr['SACC']).prod() - 1) * 100:+.1f}%"
          f"   spread {dd.mean() * 252:+.2%}/yr (t={st.hac_t(dd):+.2f})")

    print("\n=== sector fallback - PFF vs the baby-bond basket ===")
    bk_b = st.equal_weight_basket(bond, cost_bps=COST_BPS)
    pffr = px["PFF"].pct_change().reindex(bond.index).dropna()
    e_pff = (pffr - cash.reindex(pffr.index)).dropna()
    e_bb = (bk_b["net"] - cash).dropna()
    s_pff, s_bb = st.summary(e_pff), st.summary(e_bb)
    print(f"  PFF            : {fmt(s_pff)}  MaxDD {s_pff['max_drawdown']:+.1%}")
    print(f"  baby-bond bskt : {fmt(s_bb)}  MaxDD {s_bb['max_drawdown']:+.1%}")
    gap = (e_pff - e_bb).dropna()
    print(f"  PFF - baby bonds: {gap.mean() * 252:+.2%}/yr (HAC t = {st.hac_t(gap):+.2f})")
    bk_p = st.equal_weight_basket(pref, cost_bps=COST_BPS)
    print(f"  corr(our pref ladder, PFF) = {bk_p['net'].corr(pffr):.2f}   "
          f"corr(bond ladder, PFF) = {bk_b['net'].corr(pffr):.2f}")

    print("\n=== synthetic control (machinery proof, never supports the stamp) ===")
    for ss in (1.0, 0.0):
        ts, sp = [], []
        for k in range(20):
            p_, b_, c_, _ = data.synthetic_panel(signal_strength=ss, seed=933 + k)
            d = st.synthetic_detect(p_, b_, c_)
            ts.append(d["t_pairwise"]); sp.append(d["pairwise_spread_ann"])
        ts, sp = np.array(ts), np.array(sp)
        tag = "planted +6.0%/yr" if ss else "exact null"
        print(f"  {tag:17s}: spread {sp.mean():+.2%}/yr (sd {sp.std(ddof=1):.2%})"
              f"  |t|>=2 in {int((np.abs(ts) >= 2).sum())}/20")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
