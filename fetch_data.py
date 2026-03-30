import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "data", "ind_nifty500list (1).csv")
OUT_PATH  = os.path.join(BASE_DIR, "data", "corr_data.json")

# ── load stock list ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["Symbol"] = df["Symbol"].str.strip()
df["Industry"] = df["Industry"].str.strip()
df["Company Name"] = df["Company Name"].str.strip()

stocks = df[["Company Name", "Industry", "Symbol"]].dropna().to_dict("records")
print(f"Loaded {len(stocks)} stocks from CSV")

# ── fetch 1-year prices from Yahoo Finance ─────────────────────────────────
end   = datetime.today()
start = end - timedelta(days=370)          # a little extra for alignment

tickers = [s["Symbol"] + ".NS" for s in stocks]

print(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
raw = yf.download(
    tickers,
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=True,
)

# ── extract adjusted close ─────────────────────────────────────────────────
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"]
else:
    close = raw[["Close"]]

# Drop columns with > 30% missing
thresh = int(len(close) * 0.70)
close  = close.dropna(axis=1, thresh=thresh)

# Keep only last 252 rows (1 trading year)
close  = close.tail(252)

# Forward-fill then drop rows that are still all-NaN
close  = close.ffill().dropna(how="all")

print(f"Price matrix shape after cleaning: {close.shape}")

# ── compute log returns ────────────────────────────────────────────────────
returns = np.log(close / close.shift(1)).dropna()

# ── compute full correlation matrix ───────────────────────────────────────
corr = returns.corr()

# ── build symbol metadata (only for symbols with data) ────────────────────
available_tickers = list(corr.columns)  # e.g. "TCS.NS"
available_syms    = [t.replace(".NS", "") for t in available_tickers]

sym_to_meta = {s["Symbol"]: s for s in stocks}

symbols_meta = []
for sym in available_syms:
    meta = sym_to_meta.get(sym, {"Company Name": sym, "Industry": "Unknown", "Symbol": sym})
    symbols_meta.append({
        "symbol":   sym,
        "name":     meta["Company Name"],
        "industry": meta["Industry"],
    })

# ── compute block model parameters ────────────────────────────────────────
industries = [m["industry"] for m in symbols_meta]
ind_list   = sorted(set(industries))
N          = len(symbols_meta)

corr_arr = corr.values.astype(float)

# within-industry avg rho per sector
rho_within = {}
rho_cnt    = {}
rho0_sum   = 0.0
rho0_cnt   = 0

for i in range(N):
    for j in range(i + 1, N):
        v = corr_arr[i, j]
        if np.isnan(v):
            continue
        if industries[i] == industries[j]:
            key = industries[i]
            rho_within[key] = rho_within.get(key, 0.0) + v
            rho_cnt[key]    = rho_cnt.get(key, 0) + 1
        else:
            rho0_sum += v
            rho0_cnt += 1

rho_within_avg = {k: rho_within[k] / rho_cnt[k] for k in rho_within if rho_cnt[k] > 0}
rho0           = rho0_sum / rho0_cnt if rho0_cnt > 0 else 0.0

# grand avg rho
all_off = [corr_arr[i, j] for i in range(N) for j in range(i+1, N) if not np.isnan(corr_arr[i, j])]
rho_grand = float(np.mean(all_off)) if all_off else 0.0

# ── serialise correlation matrix (upper triangle only to save space) ────────
# Store as flat list row-by-row, NaN → null
corr_list = []
for row in corr_arr:
    corr_list.append([None if np.isnan(v) else round(float(v), 5) for v in row])

# ── assemble output JSON ───────────────────────────────────────────────────
out = {
    "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "trading_days":  int(len(returns)),
    "n_stocks":      N,
    "rho_grand":     round(rho_grand, 6),
    "rho0":          round(rho0, 6),
    "rho_within":    {k: round(v, 6) for k, v in rho_within_avg.items()},
    "industry_list": ind_list,
    "symbols":       symbols_meta,
    "corr_matrix":   corr_list,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f, separators=(",", ":"))

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"Saved {OUT_PATH}  ({size_kb:.0f} KB)")
print(f"Stocks: {N}  |  Trading days: {len(returns)}  |  rho_grand={rho_grand:.4f}  |  rho0={rho0:.4f}")
