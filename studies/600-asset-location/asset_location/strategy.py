"""Study 600 — Asset Location: the household simulator + inference.

THE CLAIM. *"Put the bonds in the tax-deferred account and the stocks in taxable — it adds
real after-tax return for free."* The textbook asset-location rule (Dammon-Spatt-Zhang 2004):
bond coupons are taxed every year at the ordinary-income rate, while equities pay lightly-taxed
qualified dividends and can defer capital gains — so the tax shelter should hold the
tax-*inefficient* asset.

THE MACHINE. A 60/40 household with a 50/50 taxable/traditional-IRA split, simulated one
calendar year at a time over a 30-year horizon under three **location policies** holding the
household's pre-tax 60/40 fixed:

  * ``bonds_ira``  — the IRA is filled with bonds first (the rule under test);
  * ``stocks_ira`` — the IRA is filled with stocks first (the anti-rule);
  * ``pro_rata``   — every account holds 60/40 (the no-thought default).

Taxable-account frictions each year: qualified dividends taxed at the LTCG rate, bond coupons
at the ordinary rate, a ``turnover`` fraction of the stock sleeve realised at LTCG (average-cost
basis, realised losses carried forward), bond-fund price gains realised annually. The IRA pays
nothing along the way and is taxed once at the ordinary rate on terminal withdrawal; the taxable
account settles LTCG on its unrealised gain at the end. Rebalancing back to 60/40 happens once a
year, inside the IRA for free where possible, in the taxable account (with tax consequences)
otherwise. **One documented decision lag**: the bond coupon for year *t* is the 10y yield of the
prior December — the yield the household could actually see when it allocated.

INFERENCE. Cohort = a 30-year start year. Overlapping cohorts are massively serially dependent,
so the mean after-tax delta carries a Newey-West (HAC) *t* with **lag = horizon - 1 = 29**, and
the delta-on-yield regression uses HAC standard errors at the same lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Baseline household (documented in docs/results.md)
ORDINARY = 0.24        # marginal ordinary-income rate (24% bracket)
LTCG = 0.15            # long-term capital gains + qualified dividend rate
TURNOVER = 0.10        # annual realised fraction of the taxable stock sleeve
BOND_TURNOVER = 1.00   # bond-fund price gains are realised annually (constant-maturity ladder)
SPLIT = 0.50           # taxable share of initial wealth (rest = traditional IRA)
STOCK_W = 0.60         # household equity weight (pre-tax dollars)
HORIZON = 30           # years per cohort
HAC_LAGS = HORIZON - 1  # overlap of adjacent annual-start cohorts

POLICIES = ("bonds_ira", "stocks_ira", "pro_rata")


# --------------------------------------------------------------------------- #
# The household simulator
# --------------------------------------------------------------------------- #
def _target_location(total: float, ira_total: float, tax_total: float,
                     stock_w: float, policy: str) -> tuple[float, float]:
    """Target (ira_stock, tax_stock) dollars for the policy, household stock_w fixed."""
    stock_target = stock_w * total
    bond_target = total - stock_target
    if policy == "bonds_ira":
        ira_bond = min(bond_target, ira_total)
        ira_stock = ira_total - ira_bond
    elif policy == "stocks_ira":
        ira_stock = min(stock_target, ira_total)
    elif policy == "pro_rata":
        ira_stock = stock_w * ira_total
    else:
        raise ValueError(policy)
    tax_stock = min(max(stock_target - ira_stock, 0.0), tax_total)
    return ira_stock, tax_stock


def simulate_household(comp: pd.DataFrame, policy: str, *,
                       ordinary: float = ORDINARY, ltcg: float = LTCG,
                       turnover: float = TURNOVER, bond_turnover: float = BOND_TURNOVER,
                       split: float = SPLIT, stock_w: float = STOCK_W) -> dict:
    """Run one household over the annual components frame ``comp`` under ``policy``.

    Returns pre-tax and after-tax terminal wealth (initial wealth = 1.0 pre-tax dollars),
    the annualised after-tax return, and the average annual taxable trade volume (% of NAV)
    so the desk can verify the policies trade comparable amounts.
    """
    n = len(comp)
    ira_total = 1.0 - split
    tax_total = split
    ira_s, tax_s = _target_location(1.0, ira_total, tax_total, stock_w, policy)
    ira_b = ira_total - ira_s
    tax_b = tax_total - tax_s
    basis_s, basis_b = tax_s, tax_b          # average-cost basis of the taxable sleeves
    loss_carry = 0.0                          # realised-loss carryforward (taxable)
    trade_vol = 0.0                           # cumulative taxable trades (for the cost note)

    def _sell(value: float, basis: float, frac: float, rate: float,
              loss: float) -> tuple[float, float, float, float]:
        """Sell ``frac`` of a taxable sleeve at average cost; returns
        (net proceeds, new value, new basis, new loss_carry)."""
        proceeds = frac * value
        sold_basis = frac * basis
        realised = proceeds - sold_basis
        tax = 0.0
        if realised > 0:
            offset = min(loss, realised)
            loss -= offset
            tax = rate * (realised - offset)
        else:
            loss += -realised
        return proceeds - tax, (1.0 - frac) * value, (1.0 - frac) * basis, loss

    for _, row in comp.iterrows():
        g, d, c, bp = row["stock_px"], row["stock_dy"], row["bond_cpn"], row["bond_px"]

        # ---- grow the IRA (no tax, income reinvested in place) -------------
        ira_s *= (1.0 + g + d)
        ira_b *= (1.0 + bp + c)

        # ---- grow the taxable account (income taxed, net reinvested) -------
        div_cash = d * tax_s
        tax_s *= (1.0 + g)
        net_div = div_cash * (1.0 - ltcg)             # qualified-dividend rate
        tax_s += net_div
        basis_s += net_div

        cpn_cash = c * tax_b
        tax_b *= (1.0 + bp)
        net_cpn = cpn_cash * (1.0 - ordinary)         # coupons at the ordinary rate
        tax_b += net_cpn
        basis_b += net_cpn

        # ---- annual realisation (fund turnover), sell-and-rebuy ------------
        if turnover > 0 and tax_s > 0:
            net, tax_s, basis_s, loss_carry = _sell(tax_s, basis_s, turnover, ltcg, loss_carry)
            tax_s += net
            basis_s += net
        if bond_turnover > 0 and tax_b > 0:
            net, tax_b, basis_b, loss_carry = _sell(tax_b, basis_b, bond_turnover, ltcg, loss_carry)
            tax_b += net
            basis_b += net

        # ---- year-end relocation back to the policy's 60/40 ----------------
        ira_total = ira_s + ira_b
        tax_total = tax_s + tax_b
        total = ira_total + tax_total
        t_ira_s, t_tax_s = _target_location(total, ira_total, tax_total, stock_w, policy)
        ira_s, ira_b = t_ira_s, ira_total - t_ira_s   # IRA trades are free

        if t_tax_s < tax_s - 1e-12:                    # sell taxable stock, buy bonds
            frac = (tax_s - t_tax_s) / tax_s
            trade_vol += 2.0 * (tax_s - t_tax_s) / total
            net, tax_s, basis_s, loss_carry = _sell(tax_s, basis_s, frac, ltcg, loss_carry)
            tax_b += net
            basis_b += net
        elif t_tax_s > tax_s + 1e-12 and tax_b > 0:    # sell taxable bonds, buy stock
            need = min(t_tax_s - tax_s, tax_b)
            frac = need / tax_b
            trade_vol += 2.0 * need / total
            net, tax_b, basis_b, loss_carry = _sell(tax_b, basis_b, frac, ltcg, loss_carry)
            tax_s += net
            basis_s += net

    # ---- terminal settlement -------------------------------------------------
    pre_tax = ira_s + ira_b + tax_s + tax_b
    gains = (tax_s - basis_s) + (tax_b - basis_b) - loss_carry
    tax_due = ltcg * max(gains, 0.0)
    after_tax = (tax_s + tax_b - tax_due) + (ira_s + ira_b) * (1.0 - ordinary)
    return {
        "pre_tax": float(pre_tax),
        "after_tax": float(after_tax),
        "ann_after_tax": float(after_tax ** (1.0 / n) - 1.0),
        "avg_taxable_trade_pct": float(100.0 * trade_vol / n),
    }


# --------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------- #
def cohort_table(comp: pd.DataFrame, horizon: int = HORIZON, **kw) -> pd.DataFrame:
    """Every ``horizon``-year cohort: after-tax annualised return per policy + deltas (bps/yr).

    ``delta_bs`` = bonds_ira - stocks_ira; ``delta_bp`` = bonds_ira - pro_rata (both bps/yr of
    after-tax annualised return). ``mean_cpn`` is the cohort's average locked 10y coupon.
    """
    years = comp.index.to_numpy()
    rows = []
    for i in range(0, len(years) - horizon + 1):
        window = comp.iloc[i:i + horizon]
        res = {p: simulate_household(window, p, **kw) for p in POLICIES}
        rows.append({
            "start": int(years[i]),
            "ann_bonds_ira": res["bonds_ira"]["ann_after_tax"],
            "ann_stocks_ira": res["stocks_ira"]["ann_after_tax"],
            "ann_pro_rata": res["pro_rata"]["ann_after_tax"],
            "wealth_bonds_ira": res["bonds_ira"]["after_tax"],
            "wealth_stocks_ira": res["stocks_ira"]["after_tax"],
            "wealth_pro_rata": res["pro_rata"]["after_tax"],
            "delta_bs": 1e4 * (res["bonds_ira"]["ann_after_tax"] - res["stocks_ira"]["ann_after_tax"]),
            "delta_bp": 1e4 * (res["bonds_ira"]["ann_after_tax"] - res["pro_rata"]["ann_after_tax"]),
            "mean_cpn": float(window["bond_cpn"].mean()),
            "trade_bonds_ira": res["bonds_ira"]["avg_taxable_trade_pct"],
            "trade_stocks_ira": res["stocks_ira"]["avg_taxable_trade_pct"],
            "trade_pro_rata": res["pro_rata"]["avg_taxable_trade_pct"],
        })
    return pd.DataFrame(rows).set_index("start")


# --------------------------------------------------------------------------- #
# Inference — HAC t on overlapping cohorts
# --------------------------------------------------------------------------- #
def hac_tstat(x: np.ndarray, lags: int = HAC_LAGS) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def hac_ols(y: np.ndarray, x: np.ndarray, lags: int = HAC_LAGS) -> dict:
    """OLS of y on [1, x] with Newey-West standard errors; returns slope + its t."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(y) & np.isfinite(x)
    y, x = y[keep], x[keep]
    n = y.size
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    Z = X * e[:, None]
    S = (Z.T @ Z) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        G = (Z[k:].T @ Z[:-k]) / n
        S += w * (G + G.T)
    cov = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov) / n)
    return {"slope": float(beta[1]), "t_slope": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
            "intercept": float(beta[0])}


def summarize_cohorts(tab: pd.DataFrame) -> dict:
    """Headline stats of the cohort table: mean deltas, HAC t (lag 29), win rates,
    terminal-wealth gap, and the delta-on-yield regression."""
    d_bs = tab["delta_bs"].to_numpy()
    d_bp = tab["delta_bp"].to_numpy()
    reg = hac_ols(d_bs, 100.0 * tab["mean_cpn"].to_numpy())   # bps/yr per 1pp of coupon
    wgap = 100.0 * (tab["wealth_bonds_ira"] / tab["wealth_stocks_ira"] - 1.0)
    return {
        "n_cohorts": int(len(tab)),
        "mean_delta_bs": float(d_bs.mean()), "t_bs": hac_tstat(d_bs),
        "win_bs": float((d_bs > 0).mean() * 100.0),
        "mean_delta_bp": float(d_bp.mean()), "t_bp": hac_tstat(d_bp),
        "win_bp": float((d_bp > 0).mean() * 100.0),
        "min_delta_bs": float(d_bs.min()), "max_delta_bs": float(d_bs.max()),
        "mean_wealth_gap_pct": float(wgap.mean()),
        "slope_bps_per_pp": reg["slope"], "t_slope": reg["t_slope"],
        "mean_ann_bonds_ira": float(tab["ann_bonds_ira"].mean() * 100.0),
        "mean_ann_stocks_ira": float(tab["ann_stocks_ira"].mean() * 100.0),
        "mean_ann_pro_rata": float(tab["ann_pro_rata"].mean() * 100.0),
        "mean_trade_bonds_ira": float(tab["trade_bonds_ira"].mean()),
        "mean_trade_stocks_ira": float(tab["trade_stocks_ira"].mean()),
        "mean_trade_pro_rata": float(tab["trade_pro_rata"].mean()),
    }


def yield_quartiles(tab: pd.DataFrame) -> pd.DataFrame:
    """Mean delta (bonds-in-IRA vs stocks-in-IRA) by cohort-average-coupon quartile."""
    t = tab.copy()
    t["q"] = pd.qcut(t["mean_cpn"], 4, labels=["Q1 (lowest yields)", "Q2", "Q3", "Q4 (highest)"])
    g = t.groupby("q", observed=True).agg(
        mean_cpn_pct=("mean_cpn", lambda s: 100.0 * s.mean()),
        mean_delta_bs=("delta_bs", "mean"),
        n=("delta_bs", "size"),
    )
    return g


def rate_grid(comp: pd.DataFrame, ordinaries=(0.22, 0.24, 0.32, 0.37),
              ltcgs=(0.15, 0.20), **kw) -> pd.DataFrame:
    """Mean bonds-in-IRA-vs-stocks-in-IRA delta (bps/yr) + HAC t across the tax-rate grid."""
    rows = []
    for o in ordinaries:
        for lt in ltcgs:
            tab = cohort_table(comp, ordinary=o, ltcg=lt, **kw)
            rows.append({"ordinary": o, "ltcg": lt,
                         "mean_delta_bs": float(tab["delta_bs"].mean()),
                         "t_bs": hac_tstat(tab["delta_bs"].to_numpy()),
                         "win_pct": float((tab["delta_bs"] > 0).mean() * 100.0)})
    return pd.DataFrame(rows)


def turnover_flip(comp: pd.DataFrame, turnovers=(0.0, 0.10, 0.25, 0.50, 1.00),
                  **kw) -> pd.DataFrame:
    """Third axis: sweep the equity-turnover knob and watch for the sign flip,
    overall and within the lowest-yield-quartile cohorts."""
    rows = []
    for tv in turnovers:
        tab = cohort_table(comp, turnover=tv, **kw)
        lo = tab[tab["mean_cpn"] <= tab["mean_cpn"].quantile(0.25)]
        rows.append({"turnover": tv,
                     "mean_delta_bs": float(tab["delta_bs"].mean()),
                     "t_bs": hac_tstat(tab["delta_bs"].to_numpy()),
                     "win_pct": float((tab["delta_bs"] > 0).mean() * 100.0),
                     "lowq_delta_bs": float(lo["delta_bs"].mean())})
    return pd.DataFrame(rows)


def single_cohort(comp: pd.DataFrame, **kw) -> dict:
    """One cohort over the full ``comp`` window (used for the modern SPY/IEF tape)."""
    res = {p: simulate_household(comp, p, **kw) for p in POLICIES}
    return {
        "n_years": int(len(comp)),
        "mean_cpn_pct": float(100.0 * comp["bond_cpn"].mean()),
        "ann_bonds_ira": float(100.0 * res["bonds_ira"]["ann_after_tax"]),
        "ann_stocks_ira": float(100.0 * res["stocks_ira"]["ann_after_tax"]),
        "ann_pro_rata": float(100.0 * res["pro_rata"]["ann_after_tax"]),
        "delta_bs": float(1e4 * (res["bonds_ira"]["ann_after_tax"] - res["stocks_ira"]["ann_after_tax"])),
        "delta_bp": float(1e4 * (res["bonds_ira"]["ann_after_tax"] - res["pro_rata"]["ann_after_tax"])),
    }


def synthetic_scaling(coupons=(0.02, 0.04, 0.06), n_seeds: int = 20,
                      **kw) -> pd.DataFrame:
    """Positive control, averaged over >= 20 seeds (no single-seed baseline).

    * **The null** — the same seeded worlds run with **zero tax rates**: with no taxes the
      account boundary is meaningless, so every location policy must return *exactly* the same
      after-tax wealth (delta = 0.0 to machine precision). A simulator that manufactures a
      location benefit out of nothing fails here.
    * **The planted effect** — the bond ``coupon`` knob. The whole mechanism is annually-taxed
      income sheltered by the IRA, so the measured delta must scale monotonically with the
      coupon. Same world settings on every row; only the knob moves.
    """
    from . import data as _data

    rows = []
    # the exact null: zero tax rates -> location cannot matter
    null_deltas = []
    for s in range(n_seeds):
        comp = _data.synthetic_world(coupon=0.05, seed=600 + s)
        r = single_cohort(comp, ordinary=0.0, ltcg=0.0)
        null_deltas.append(r["delta_bs"])
    null_deltas = np.asarray(null_deltas)
    rows.append({"label": "null (zero tax rates)", "coupon_pct": 5.0,
                 "mean_delta_bs": float(null_deltas.mean()),
                 "sd_delta_bs": float(null_deltas.std(ddof=1)), "n_seeds": n_seeds})
    # the planted knob: coupon level
    for cpn in coupons:
        deltas = []
        for s in range(n_seeds):
            comp = _data.synthetic_world(coupon=cpn, seed=600 + s)
            deltas.append(single_cohort(comp, **kw)["delta_bs"])
        deltas = np.asarray(deltas)
        rows.append({"label": f"planted coupon {100.0 * cpn:.0f}%",
                     "coupon_pct": 100.0 * cpn, "mean_delta_bs": float(deltas.mean()),
                     "sd_delta_bs": float(deltas.std(ddof=1)), "n_seeds": n_seeds})
    return pd.DataFrame(rows)
