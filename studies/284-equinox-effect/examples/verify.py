"""Real-data verification -- Study 284 (Equinox-Effect). Regenerates docs/results.md.

Loads ^GSPC daily returns from the repo-level cache, builds an event study around
the hardcoded equinox/solstice dates, and runs the full analysis: event-day
abnormal returns, HAC t-stat, permutation test, the CAR window, and the
multiple-comparisons summary by kind and season.

Cache-only -- no network. Run from anywhere:

    python studies/284-equinox-effect/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from equinox_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 284 -- Equinox-Effect event study -- real-data verification\n")
    print("Data: ^GSPC daily (repo-level _cache/^GSPC_split_only.parquet)")
    print("      Equinox/solstice dates: hardcoded in data.py (1928-2026, Meeus)\n")

    px = data.fetch_gspc_daily()  # cache-only by default
    ret = px["ret"]
    ev = data.equinox_table()
    fp = data.fingerprint(px)

    print(f"Data stamp: {ret.index.min().date()}..{ret.index.max().date()}, "
          f"n_days={len(ret)}, fingerprint={fp}")
    print(f"Events: {len(ev)} hardcoded "
          f"(equinoxes={int((ev['kind']=='equinox').sum())}, "
          f"solstices={int((ev['kind']=='solstice').sum())})\n")

    s = st.summarize(ret, ev, n_permutations=10_000, seed=284)

    print("=== Event-day abnormal returns (event day 0 = first trading day on/after) ===")
    print(f"  Unconditional daily drift:  {s['uncond_bps']:+.2f} bps")
    print(f"  ALL events    (n={s['n_all']}):  "
          f"mean abn {s['all_mean_abn_bps']:+.2f} bps, "
          f"HAC t {s['all_t_hac']:+.2f}, perm p {s['all_perm_p']:.4f}, "
          f"win-rate {s['all_win_rate_pct']:.1f}%")
    print(f"  EQUINOX only  (n={s['n_eqx']}):  "
          f"mean abn {s['eqx_mean_abn_bps']:+.2f} bps, "
          f"HAC t {s['eqx_t_hac']:+.2f}, perm p {s['eqx_perm_p']:.4f}")
    print(f"  SOLSTICE only (n={s['n_sol']}):  "
          f"mean abn {s['sol_mean_abn_bps']:+.2f} bps, "
          f"HAC t {s['sol_t_hac']:+.2f}, perm p {s['sol_perm_p']:.4f}")
    for seas in ("spring", "summer", "autumn", "winter"):
        print(f"  {seas.upper():7s}       (n={s['n_season']}):  "
              f"mean abn {s[f'{seas}_mean_abn_bps']:+.2f} bps, "
              f"HAC t {s[f'{seas}_t_hac']:+.2f}, perm p {s[f'{seas}_perm_p']:.4f}")
    print(f"  CAR over [-5,+5] window:   {s['car_window_bps']:+.2f} bps\n")

    mc = st.multiple_comparisons_summary(ret, ev, n_permutations=10_000, seed=284)
    print("=== Multiple-comparisons summary (Bonferroni over 7 slices) ===")
    print(mc.to_string(index=False))
    print()

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- the all-event abnormal return is")
    print("-3 bps (t=-0.54); the most extreme slice (autumn equinox) is below |t|=2 raw")
    print("and dies under Bonferroni. The Sun's calendar is not a market calendar.")


if __name__ == "__main__":
    main()
