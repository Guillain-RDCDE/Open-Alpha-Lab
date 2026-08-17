"""Real-tape verification — Study 929 (Rights Offering). Regenerates docs/results.md.

Reads the shared cache (twenty US rights issuers, SPY and BIL), runs the market-model
event study across the announcement, subscription and post-expiry windows, then the
whole robustness battery: bootstrap CIs, leave-one-out and leave-one-issuer-out
jackknives, three placebos (whole-tape, era-matched and clustered), the discount-band
split with a label permutation, an era cut, an anchor jitter, a timetable sweep, and the
costed calendar-time portfolio — long and short, over the full and the ex-rights-free
holding span, with a beta-adjusted alpha and a borrow sweep. Network only on ``--fetch``.

    python studies/929-rights-offering-discount/examples/verify.py            # cache-only
    python studies/929-rights-offering-discount/examples/verify.py --fetch    # refresh
    python studies/929-rights-offering-discount/examples/verify.py --quick    # skip placebo
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rights_offering import data, strategy as st  # noqa: E402

HOLD = (1, 49)          # enter the session AFTER the announcement, exit 49 sessions on
HOLD_CLEAN = (29, 49)   # the ex-rights-free span: no mechanical dilution in either leg
COST_BPS = 25.0         # one-way, x NAV — small, wide-spread closed-end funds
BORROW_BPS = 300.0      # annualised borrow ASSUMPTION for the short leg (swept below)
# Era-matched placebo range: pseudo-anchors drawn only from the years in which deals on
# this list were actually announced (2013-05 .. 2023-09), padded by a year each side.
# Drawing from the whole 2005-2026 tape hands the placebo the 2008 and 2020 crashes and
# inflates its dispersion — a yardstick the real deals never faced.
ERA_LO, ERA_HI = "2012-01-01", "2024-06-30"


def synthetic_demo() -> None:
    """Offline fallback: run the same machinery on the planted and null synthetic panels.

    Printed instead of the real-tape tables when the shared cache is absent (a fresh
    checkout), so this script is always runnable and always exits 0. These numbers are
    synthetic and never support a verdict.
    """
    print("No shared _cache found — running the OFFLINE SYNTHETIC demo instead.")
    print("(run with --fetch once to populate studies/_cache and get the real tape)\n")
    for ss in (1.0, 0.0):
        sp, sev, truth = data.synthetic_panel(signal_strength=ss, seed=929)
        panel = st.event_panel(sp, sev)
        summ = st.event_summary(panel)
        print(f"signal_strength={ss:.0f}  ({truth['n_events']} planted deals, "
              f"{truth['n_names']} names)")
        for name in ("announce", "subscription", "post_expiry"):
            s = summ[name]
            print(f"   {name:13s} mean {s['mean_pct']:+6.2f}%  t={s['t']:+6.2f}  "
                  f"hit {s['hit_rate']:.0%}")
        d = st.discount_split(panel, "post_expiry")
        print(f"   deep - rest (post-expiry): {d['diff_pct']:+.2f}% "
              f"(Welch t={d['welch_t']:+.2f})\n")


def main(fetch: bool, quick: bool) -> None:
    if fetch:
        data.fetch()
    if not data.have_real():
        synthetic_demo()
        return
    px = data.load_prices()
    ev = data.RIGHTS_EVENTS

    cov = st.coverage(px, ev)
    print(f"tape {px.index[0].date()} -> {px.index[-1].date()}  n={len(px)}  "
          f"fp={data.fingerprint(px)}")
    print(f"as-of {data.AS_OF}")
    print(f"events listed {cov['n_listed']}  usable {cov['n_usable']}  "
          f"tickers {cov['n_tickers']}  dropped {cov['dropped']}")
    print(f"deals with no retrievable tape at all (survivorship): "
          f"{', '.join(data.DROPPED_NO_TAPE)}")

    panel = st.event_panel(px, ev)
    summ = st.event_summary(panel)

    print("\n=== event windows (market-model CAR vs SPY, trading days from anchor) ===")
    for name, (lo, hi) in st.WINDOWS.items():
        s = summ[name]
        boot = st.bootstrap_mean_ci(panel[f"car_{name}"].to_numpy(), seed=929)
        print(f"  {name:13s} [{lo:+3d},{hi:+3d}]  n={s['n']:3d}  mean={s['mean_pct']:+6.2f}%  "
              f"med={s['median_pct']:+6.2f}%  t={s['t']:+5.2f}  hit={s['hit_rate']:.0%} "
              f"[{s['hit_lo']:.0%},{s['hit_hi']:.0%}]  boot95 [{boot['ci_low']:+.2f}%, "
              f"{boot['ci_high']:+.2f}%]")

    print("\n=== jackknives (the list is dominated by serial issuers) ===")
    for w in ("subscription", "post_expiry"):
        j = st.jackknife(panel, w)
        print(f"  {w:13s} leave-one-out: mean in [{j['min_mean_pct']:+.2f}%, "
              f"{j['max_mean_pct']:+.2f}%], t in [{j['min_t']:+.2f}, {j['max_t']:+.2f}]")
        ji = st.jackknife_issuer(panel, w)
        worst = ji.iloc[-1]
        best = ji.iloc[0]
        print(f"                leave-one-ISSUER-out: t ranges {best['t']:+.2f} "
              f"(drop {best['dropped']}) to {worst['t']:+.2f} (drop {worst['dropped']})")

    print("\n=== discount band: is the deep discount compensated? ===")
    for w in ("subscription", "post_expiry"):
        d = st.discount_split(panel, w)
        perm = st.permutation_discount_test(panel, w, seed=929)
        print(f"  {w:13s} deep (n={d['n_deep']}) {d['mean_deep_pct']:+.2f}%  vs  "
              f"rest (n={d['n_rest']}) {d['mean_rest_pct']:+.2f}%  "
              f"gap {d['diff_pct']:+.2f}% (Welch t={d['welch_t']:+.2f}; "
              f"label-permutation p={perm['p_two_sided']:.3f})")

    print("\n=== era cut (split 2018-01-01) ===")
    for w in ("subscription", "post_expiry"):
        eras = st.era_cut(px, ev, split="2018-01-01", window=w)
        bits = []
        for tag, e in eras.items():
            bits.append(f"{tag} n={e['n']} {e['mean_pct']:+.2f}% (t={e['t']:+.2f})"
                        if e else f"{tag} n/a")
        print(f"  {w:13s} " + "   ".join(bits))

    print("\n=== anchor jitter (the list is month-precision) ===")
    for w in ("announce", "subscription", "post_expiry"):
        rows = st.anchor_jitter(px, ev, window=w)
        print(f"  {w:13s} " + "  ".join(
            f"{r['shift_days']:+3d}d {r['mean_pct']:+5.2f}%(t={r['t']:+.2f})" for r in rows))

    print("\n=== assumed timetable sweep (expiry days from announcement) ===")
    for r in st.timetable_sweep(px, ev):
        print(f"  expiry={r['expiry_days']:3d}d  subscription {r['subscription_mean_pct']:+6.2f}% "
              f"(t={r['subscription_t']:+.2f})   post-expiry {r['post_expiry_mean_pct']:+6.2f}% "
              f"(t={r['post_expiry_t']:+.2f})")

    if not quick:
        print("\n=== placebo A: random pseudo-anchors, drawn from the WHOLE tape ===")
        print("    (2005-2026 — includes 2008 and 2020, regimes no deal here lived in:")
        print("     this version FLATTERS the null, so placebo B is the one to quote)")
        for w in ("announce", "subscription", "post_expiry"):
            pl = st.placebo(px, ev, window=w, n_draws=300, seed=929)
            print(f"  {w:13s} observed {pl['observed_pct']:+.2f}%  vs placebo "
                  f"{pl['placebo_mean_pct']:+.2f}% (sd {pl['placebo_sd_pct']:.2f}%)  "
                  f"z={pl['z']:+.2f}  p={pl['p_two_sided']:.3f}")

        print(f"\n=== placebo B: ERA-MATCHED pseudo-anchors ({ERA_LO} .. {ERA_HI}) ===")
        for w in ("announce", "subscription", "post_expiry"):
            pl = st.placebo(px, ev, window=w, n_draws=300, seed=929,
                            lo_date=ERA_LO, hi_date=ERA_HI)
            print(f"  {w:13s} observed {pl['observed_pct']:+.2f}%  vs placebo "
                  f"{pl['placebo_mean_pct']:+.2f}% (sd {pl['placebo_sd_pct']:.2f}%)  "
                  f"z={pl['z']:+.2f}  p={pl['p_two_sided']:.3f}")

        print("\n=== placebo C: CLUSTERED — slide the whole list by one random offset ===")
        print("    (keeps every gap between deals, so the cross-event dependence stays)")
        for w in ("announce", "subscription", "post_expiry"):
            pl = st.clustered_placebo(px, ev, window=w, n_draws=300, seed=929)
            print(f"  {w:13s} observed {pl['observed_pct']:+.2f}%  vs placebo "
                  f"{pl['placebo_mean_pct']:+.2f}% (sd {pl['placebo_sd_pct']:.2f}%)  "
                  f"z={pl['z']:+.2f}  p={pl['p_two_sided']:.3f}  ({pl['n_draws']} draws)")

    print("\n=== tradability: calendar-time equal-weight portfolio, excess-of-cash ===")
    print("    both legs excess-of-cash; the short leg is credited its cash collateral")
    print("    and pays borrow on top. Alpha = beta-adjusted vs SPY (the long book rents")
    print("    equity beta a quarter of the time, so its OWN t is not evidence of edge).")
    for hold, htag in ((HOLD, "full (+1,+49) — dilution INSIDE"),
                       (HOLD_CLEAN, "clean (+29,+49) — ex-rights free")):
        print(f"  -- holding {htag}")
        for side, tag in ((1, "long (discount = gift)"), (-1, "short (discount = warning)")):
            t = st.tradability(px, ev, hold=hold, cost_bps=COST_BPS, side=side,
                               borrow_bps_ann=BORROW_BPS if side < 0 else 0.0)
            p, a = t["port"], t["alpha"]
            print(f"     {tag:26s} exSharpe {p['sharpe']:+.3f}  own HAC t {t['t_port']:+.2f}  "
                  f"alpha {a['alpha_ann_pct']:+.2f}%/yr (beta {a['beta']:+.2f}, "
                  f"HAC t {a['t_alpha']:+.2f})  CAGR {p['cagr']:+.2%}  "
                  f"DD {p['max_drawdown']:+.1%}  vs SPY adv {t['sharpe_adv']:+.3f} "
                  f"(t={t['t_diff']:+.2f})")
            if side == 1:
                print(f"     {'':26s} invested {t['invested_frac']:.1%} of days, "
                      f"avg {t['avg_n_active']:.1f} names live")

    print("\n=== cost sweep (long leg, one-way bps x NAV) ===")
    for r in st.cost_sweep(px, ev, hold=HOLD):
        print(f"  {r['cost_bps']:5.1f} bps  exSharpe {r['sharpe_port']:+.3f}  "
              f"vs-SPY adv {r['sharpe_adv']:+.3f} (t={r['t_diff']:+.2f})  "
              f"mean diff {r['mean_diff_bps']:+.2f} bps/day")

    print("\n=== borrow sweep (short leg, annualised bps; cash collateral credited) ===")
    for hold, htag in ((HOLD, "full "), (HOLD_CLEAN, "clean")):
        for r in st.borrow_sweep(px, ev, hold=hold, cost_bps=COST_BPS):
            print(f"  {htag} {r['borrow_bps_ann']:6.1f} bps/yr  "
                  f"exSharpe {r['sharpe_port']:+.3f}  own HAC t {r['t_port']:+.2f}")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    for ss in (1.0, 0.0):
        sp, sev, _ = data.synthetic_panel(signal_strength=ss, seed=929)
        d = st.synthetic_detect(sp, sev)
        print(f"  signal_strength={ss:.0f}: n={d['n']}  announce {d['announce_mean_pct']:+.2f}% "
              f"(t={d['announce_t']:+.2f})  subscription {d['subscription_mean_pct']:+.2f}% "
              f"(t={d['subscription_t']:+.2f})  post-expiry {d['post_expiry_mean_pct']:+.2f}% "
              f"(t={d['post_expiry_t']:+.2f})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv, quick="--quick" in sys.argv)
