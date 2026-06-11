"""Build the offline cache of alternative real-data sources — once and for all.

The FRED graph CSV endpoint times out from this environment, so the desk sources the same economics from
key-free mirrors that DO respond here:

  * **G10 short rates** — OECD MEI 3-month interbank (IR3TIB01) + immediate/call (IRSTCI01) via DBnomics.
  * **US Treasury curve** — Yahoo ^IRX / ^FVX / ^TNX / ^TYX (13-week … 30-year), current.
  * **US macro** — CPI (BLS CUSR0000SA0 via DBnomics), industrial production & unemployment (OECD MEI).
  * **Cross-asset ETFs** — Yahoo (equities, bonds, credit, commodities, TIPS, REIT, dollar).
  * **G10 FX spot** — Yahoo.
  * **Energy futures term structure** — EIA v2 (WTI RCLC1-4, Henry Hub gas RNGC1-4), the real roll-yield curve.
  * **Reported earnings** — SEC EDGAR companyconcept (diluted EPS + filing dates) for the S&P panel,
    the raw material for a seasonal-random-walk SUE (post-earnings-drift) with no estimate feed needed.

Run once (network):  python tools/fetch_altdata.py
Everything lands in _cache/*.parquet (gitignored); the studies then read the cache offline.
"""
from __future__ import annotations
import os, io, json, time, urllib.request, sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "_cache")
os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (OpenAlphaLab research)"}
SEC_UA = {"User-Agent": "OpenAlphaLab research guillain@poulpe.us"}


def _get(url, timeout=30, headers=None, tries=4):
    last = None
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or UA), timeout=timeout).read()
        except Exception as e:
            last = e; time.sleep(1.5)
    raise last


def dbnomics_series(provider_dataset_series: str) -> pd.Series:
    """One DBnomics series → a pandas Series indexed by period (monthly 'YYYY-MM' → month-end Timestamp)."""
    raw = _get(f"https://api.db.nomics.world/v22/series/{provider_dataset_series}?observations=1")
    d = json.loads(raw)["series"]["docs"][0]
    s = pd.Series(d["value"], index=pd.PeriodIndex(d["period"], freq="M").to_timestamp("M"))
    return s[s.notna()].astype(float)


# ---------------------------------------------------------------- G10 short rates (OECD via DBnomics)
G10 = {  # currency -> OECD country code
    "USD": "USA", "JPY": "JPN", "GBP": "GBR", "EUR": "EA19", "CAD": "CAN",
    "AUD": "AUS", "CHF": "CHE", "SEK": "SWE", "NOK": "NOR", "NZD": "NZL",
}

def fetch_g10_short_rates():
    out3, outc = {}, {}
    for ccy, c in G10.items():
        for ind, store in (("IR3TIB01.ST.M", out3), ("IRSTCI01.ST.M", outc)):
            try:
                store[ccy] = dbnomics_series(f"OECD/MEI/{c}.{ind}")
            except Exception as e:
                print(f"   short-rate {ccy} {ind}: FAIL {type(e).__name__}")
    df3 = pd.DataFrame(out3).sort_index(); dfc = pd.DataFrame(outc).sort_index()
    df3.index.name = dfc.index.name = "date"
    # prefer 3-month interbank; backfill currencies/periods it lacks with the call rate
    rates = df3.combine_first(dfc)
    rates.to_parquet(os.path.join(CACHE, "g10_short_rates.parquet"))
    print(f"   g10_short_rates: {rates.shape} {rates.index.min().date()}..{rates.index.max().date()} cols={list(rates.columns)}")


# ---------------------------------------------------------------- US Treasury curve + FX + ETFs (Yahoo)
def _yf(tickers, auto_adjust=True):
    import yfinance as yf
    h = yf.download(tickers, period="max", interval="1d", progress=False, auto_adjust=auto_adjust)
    if isinstance(h.columns, pd.MultiIndex):
        h = h["Close"]
    else:
        h = h[["Close"]]; h.columns = tickers if isinstance(tickers, str) else tickers
    h.index = pd.DatetimeIndex(h.index).tz_localize(None)
    h.index.name = "date"
    return h.dropna(how="all")

def fetch_us_treasury():
    df = _yf(["^IRX", "^FVX", "^TNX", "^TYX"], auto_adjust=False).rename(
        columns={"^IRX": "y3m", "^FVX": "y5y", "^TNX": "y10y", "^TYX": "y30y"})
    df.to_parquet(os.path.join(CACHE, "us_treasury_yields.parquet"))
    print(f"   us_treasury_yields: {df.shape} {df.index.min().date()}..{df.index.max().date()}")

FX = {"EURUSD=X": "EUR", "GBPUSD=X": "GBP", "AUDUSD=X": "AUD", "NZDUSD=X": "NZD",
      "JPY=X": "JPY", "CAD=X": "CAD", "CHF=X": "CHF", "SEK=X": "SEK", "NOK=X": "NOK"}
def fetch_g10_fx():
    raw = _yf(list(FX), auto_adjust=True)
    # normalise to USD per 1 unit of foreign currency
    usd_per = {}
    for tk, ccy in FX.items():
        if tk not in raw.columns:
            continue
        s = raw[tk].dropna()
        usd_per[ccy] = s if tk.endswith("USD=X") else (1.0 / s)
    df = pd.DataFrame(usd_per).sort_index(); df.index.name = "date"
    df.to_parquet(os.path.join(CACHE, "g10_fx.parquet"))
    print(f"   g10_fx (USD per unit): {df.shape} cols={list(df.columns)}")

ASSETS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "LQD", "HYG",
          "DBC", "GLD", "SLV", "USO", "DBA", "TIP", "UUP", "VNQ"]
def fetch_cross_asset():
    df = _yf(ASSETS, auto_adjust=True).sort_index()
    df.to_parquet(os.path.join(CACHE, "cross_asset_etfs.parquet"))
    print(f"   cross_asset_etfs: {df.shape} {df.index.min().date()}..{df.index.max().date()}")


# ---------------------------------------------------------------- US macro (BLS CPI + OECD via DBnomics)
def fetch_macro_us():
    cols = {}
    try: cols["cpi"] = dbnomics_series("BLS/cu/CUSR0000SA0")            # CPI-U, SA, current
    except Exception as e: print("   cpi FAIL", e)
    try: cols["indpro"] = dbnomics_series("OECD/MEI/USA.PRINTO01.IXOBSA.M")  # industrial production SA
    except Exception as e: print("   indpro FAIL", e)
    try: cols["unrate"] = dbnomics_series("OECD/MEI/USA.LRHUTTTT.ST.M")      # unemployment rate
    except Exception as e: print("   unrate FAIL", e)
    df = pd.DataFrame(cols).sort_index(); df.index.name = "date"
    df.to_parquet(os.path.join(CACHE, "macro_us.parquet"))
    print(f"   macro_us: {df.shape} {df.index.min().date()}..{df.index.max().date()} cols={list(df.columns)}")


# ---------------------------------------------------------------- Energy futures curve (EIA v2)
def _eia_series(route, series_id, key="DEMO_KEY"):
    rows, offset = [], 0
    while True:
        url = (f"https://api.eia.gov/v2/{route}/data/?api_key={key}&frequency=daily&data[0]=value"
               f"&facets[series][]={series_id}&sort[0][column]=period&sort[0][direction]=asc"
               f"&offset={offset}&length=5000")
        j = json.loads(_get(url))
        data = j["response"]["data"]
        if not data:
            break
        rows += [(d["period"], d["value"]) for d in data]
        if len(data) < 5000:
            break
        offset += 5000
        time.sleep(0.5)
    s = pd.Series({pd.Timestamp(p): (float(v) if v is not None else None) for p, v in rows})
    return s[s.notna()].sort_index()

def fetch_energy_curve():
    cols = {}
    spec = [("petroleum/pri/fut", "RCLC", "WTI"), ("natural-gas/pri/fut", "RNGC", "HH")]
    for route, root, tag in spec:
        for n in (1, 2, 3, 4):
            sid = f"{root}{n}"
            try:
                cols[f"{tag}{n}"] = _eia_series(route, sid)
                print(f"   EIA {sid}: {len(cols[f'{tag}{n}'])} obs")
            except Exception as e:
                print(f"   EIA {sid}: FAIL {type(e).__name__} {str(e)[:50]}")
            time.sleep(0.5)
    df = pd.DataFrame(cols).sort_index(); df.index.name = "date"
    df.to_parquet(os.path.join(CACHE, "energy_futures_curve.parquet"))
    print(f"   energy_futures_curve: {df.shape} {df.index.min().date()}..{df.index.max().date()} cols={list(df.columns)}")


# ---------------------------------------------------------------- SEC EDGAR reported EPS (for PEAD)
def fetch_edgar_eps(max_names=503):
    tkpath = os.path.join(CACHE, "_panel_tickers.json")
    panel_tks = json.load(open(tkpath)) if os.path.exists(tkpath) else []
    cmap = json.loads(_get("https://www.sec.gov/files/company_tickers.json", headers=SEC_UA))
    t2c = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in cmap.values()}
    targets = [t for t in panel_tks if t in t2c][:max_names]
    recs = []
    for i, tk in enumerate(targets):
        for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
            try:
                j = json.loads(_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{t2c[tk]}/us-gaap/{concept}.json",
                                    headers=SEC_UA, tries=2))
                for u in j["units"]["USD/shares"]:
                    if u.get("form") in ("10-Q", "10-K") and u.get("start") and u.get("end") and u.get("filed"):
                        # keep ~quarterly spans only (60-100 days)
                        span = (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days
                        if 60 <= span <= 100:
                            recs.append((tk, u["end"], u["filed"], float(u["val"])))
                break  # diluted preferred; only fall back to basic if diluted missing
            except Exception:
                continue
        if (i + 1) % 50 == 0:
            print(f"   EDGAR {i+1}/{len(targets)} names…")
        time.sleep(0.12)  # be polite to SEC (<10 req/s)
    df = pd.DataFrame(recs, columns=["ticker", "period_end", "filed", "eps"]).drop_duplicates()
    df["period_end"] = pd.to_datetime(df["period_end"]); df["filed"] = pd.to_datetime(df["filed"])
    df = df.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    df.to_parquet(os.path.join(CACHE, "edgar_eps.parquet"))
    print(f"   edgar_eps: {df.shape} names={df['ticker'].nunique()} "
          f"{df['period_end'].min().date()}..{df['period_end'].max().date()}")


if __name__ == "__main__":
    steps = [
        ("G10 short rates (OECD/DBnomics)", fetch_g10_short_rates),
        ("US Treasury curve (Yahoo)", fetch_us_treasury),
        ("G10 FX (Yahoo)", fetch_g10_fx),
        ("Cross-asset ETFs (Yahoo)", fetch_cross_asset),
        ("US macro (BLS+OECD/DBnomics)", fetch_macro_us),
        ("Energy futures curve (EIA)", fetch_energy_curve),
        ("SEC EDGAR reported EPS", fetch_edgar_eps),
    ]
    only = sys.argv[1:] or None
    for name, fn in steps:
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        print(f"[fetch] {name} …")
        try:
            fn()
        except Exception as e:
            print(f"   !! {name} FAILED: {type(e).__name__} {str(e)[:120]}")
    print("ALL DONE")
