"""LIVE paper-trade scaffold for the '5min-btc-polymarket' strategy.

Captures the ONE thing the offline backtest could not: the REAL Polymarket
CLOB ask (the price p you would actually pay) at ~2 minutes before close,
alongside the BTC move, then resolves the outcome. Logs empirical edge = w - p.

NO private key, NO wallet, NO order ever placed. Read-only public endpoints:
  - gamma-api.polymarket.com  (find the live market by slug btc-updown-5m-<unix>)
  - clob.polymarket.com/price (best ask = side=sell)
  - api.binance.com/klines    (exact window open/close, Chainlink proxy)

Polymarket resolves UP if BTC(end) >= BTC(start). Resolution source is Chainlink;
we proxy with Binance 1m bars (sub-dollar discrepancy, flagged in the report).

Usage:  python live_paper.py [minutes]      (default: run ~600 min)
Writes incrementally to live_log.csv ; inspect anytime with report_live.py.
"""
import csv, json, os, sys, time, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(__file__)
LOG = os.path.join(HERE, "live_log.csv")
RUN_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
WINDOW = 300                      # 5 minutes
CAP_LO, CAP_HI = 105, 140        # capture when seconds-left in this band (~2 min)
THR = 70.0                        # strategy's $ move threshold
FIELDS = ["window_slug", "window_start_utc", "capture_utc", "sec_left", "open_px",
          "spot_at_capture", "move_usd", "abs_move", "signal70",
          "up_ask", "down_ask", "favored_side", "favored_ask",
          "close_px", "outcome", "favored_win"]

def jget(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "pm5m-paper/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def binance_kline_open(start_unix):
    k = jget(f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=1&startTime={start_unix*1000}")
    return float(k[0][1]) if k else None          # open of the window's first minute

def binance_kline_close(end_unix):
    # close of the last minute of the window = bar with openTime = (end-60)
    k = jget(f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=1&startTime={(end_unix-60)*1000}")
    return float(k[0][4]) if k else None

def binance_spot():
    return float(jget("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")["price"])

def best_ask(token_id):
    return float(jget(f"https://clob.polymarket.com/price?token_id={token_id}&side=sell")["price"])

def event_tokens(start_unix):
    """Return (up_token, down_token) for the 5m window starting at start_unix, or None."""
    ev = jget(f"https://gamma-api.polymarket.com/events?slug=btc-updown-5m-{start_unix}")
    if not ev:
        return None
    mk = ev[0]["markets"][0]
    outs = json.loads(mk["outcomes"]) if isinstance(mk["outcomes"], str) else mk["outcomes"]
    ids = json.loads(mk["clobTokenIds"]) if isinstance(mk["clobTokenIds"], str) else mk["clobTokenIds"]
    m = dict(zip(outs, ids))
    return m.get("Up"), m.get("Down")

def ensure_header():
    if not os.path.exists(LOG):
        with open(LOG, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

def append(row):
    with open(LOG, "a", newline="") as f:
        csv.DictWriter(f, FIELDS).writerow(row)

def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def main():
    ensure_header()
    deadline = time.time() + RUN_MIN * 60
    captured = {}     # window_start -> partial row (captured at 2-min-left, awaiting resolution)
    open_cache = {}   # window_start -> open_px
    n_done = 0
    print(f"[start] running ~{RUN_MIN:.0f} min, logging to {LOG}", flush=True)
    while time.time() < deadline:
        try:
            now = time.time()
            wstart = int(now // WINDOW) * WINDOW
            wend = wstart + WINDOW
            sec_left = wend - now

            # 1) capture at ~2 min left (once per window)
            if CAP_LO <= sec_left <= CAP_HI and wstart not in captured:
                tok = event_tokens(wstart)
                if tok and tok[0] and tok[1]:
                    if wstart not in open_cache:
                        open_cache[wstart] = binance_kline_open(wstart)
                    open_px = open_cache[wstart]
                    spot = binance_spot()
                    up_ask, down_ask = best_ask(tok[0]), best_ask(tok[1])
                    move = spot - open_px
                    fav = "Up" if move >= 0 else "Down"
                    fav_ask = up_ask if fav == "Up" else down_ask
                    captured[wstart] = {
                        "window_slug": f"btc-updown-5m-{wstart}",
                        "window_start_utc": iso(wstart), "capture_utc": iso(now),
                        "sec_left": round(sec_left), "open_px": round(open_px, 2),
                        "spot_at_capture": round(spot, 2), "move_usd": round(move, 2),
                        "abs_move": round(abs(move), 2), "signal70": abs(move) >= THR,
                        "up_ask": up_ask, "down_ask": down_ask,
                        "favored_side": fav, "favored_ask": fav_ask,
                    }
                    sig = "SIGNAL" if abs(move) >= THR else "  no  "
                    print(f"[cap {iso(now)}] {captured[wstart]['window_slug']} move=${move:+7.1f} {sig} "
                          f"fav={fav} ask={fav_ask:.3f} (up {up_ask:.3f}/dn {down_ask:.3f})", flush=True)

            # 2) resolve windows whose end has passed (+12s settle margin)
            for ws in list(captured):
                if now >= ws + WINDOW + 12:
                    row = captured.pop(ws)
                    close_px = binance_kline_close(ws + WINDOW)
                    outcome = "Up" if close_px >= row["open_px"] else "Down"
                    row["close_px"] = round(close_px, 2)
                    row["outcome"] = outcome
                    row["favored_win"] = (outcome == row["favored_side"])
                    append(row)
                    n_done += 1
                    res = "WIN " if row["favored_win"] else "LOSS"
                    tag = "[SIGNAL]" if row["signal70"] else "        "
                    print(f"[res {iso(now)}] {row['window_slug']} {tag} close=${close_px:.1f} "
                          f"-> {outcome}  favored={row['favored_side']} {res}  paid={row['favored_ask']:.3f}  (#{n_done})", flush=True)
        except Exception as e:
            print(f"[warn] {type(e).__name__}: {e}", flush=True)
        time.sleep(7)
    print(f"[done] resolved {n_done} windows", flush=True)

if __name__ == "__main__":
    main()
