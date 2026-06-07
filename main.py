import sys, time, warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb


# Imports
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor


FILE1 = "data/ai_job_dataset.csv"
FILE2 = "data/ai_job_dataset1.csv"
TOP_K_SKILLS = 50
TEST_DAYS = 30
MAX_LAG = 14
MIN_TOTAL_COUNT = 10
RANDOM_STATE = 42
DIRECT_HORIZONS = [1, 7, 14, 30, 60]
N_ESTIMATORS = 300           # number of boosting rounds (no early stopping)
LEARNING_RATE = 0.05
NUM_LEAVES = 31
MIN_DATA_IN_LEAF = 20


t_start = time.time()
print("Start:", time.ctime())

def safe_read(path):
  """Read CSV or Excel file automatically."""
  try:
      return pd.read_csv(path)
  except Exception:
      return pd.read_excel(path)

def parse_dates_safe(series):
  """Parse dates while handling mixed formats."""
  s = pd.to_datetime(series, errors="coerce")
  if s.isna().any():
    s2 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    s[s.isna()] = s2[s.isna()]
  return s

def make_supervised(series: pd.Series, max_lag=MAX_LAG):
    idx = series.index
    df_feat = pd.DataFrame({"y": series})
    for L in range(1, max_lag + 1):
        df_feat[f"lag_{L}"] = series.shift(L)
    df_feat["roll7"]  = series.shift(1).rolling(7, min_periods=1).mean()
    df_feat["roll14"] = series.shift(1).rolling(14, min_periods=1).mean()
    df_feat["roll28"] = series.shift(1).rolling(28, min_periods=1).mean()
    df_feat["roll7_std"] = series.shift(1).rolling(7, min_periods=1).std().fillna(0.0)
    df_feat["dayofweek"] = idx.dayofweek
    df_feat["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df_feat["month"] = idx.month
    df_feat["day"] = idx.day
    for k in range(1, 4):
        df_feat[f"fourier_sin_w{k}"] = np.sin(2 * np.pi * k * (idx.dayofweek / 7))
        df_feat[f"fourier_cos_w{k}"] = np.cos(2 * np.pi * k * (idx.dayofweek / 7))
    df_feat["day_index"] = (idx - idx.min()).days
    df_feat = df_feat.dropna()
    X = df_feat.drop(columns=["y"])
    y = df_feat["y"]
    return X, y


def build_direct_row_from_history(history_series, max_lag=MAX_LAG):
    row = {}
    s = history_series
    for L in range(1, max_lag + 1):
        row[f"lag_{L}"] = s.iloc[-L] if len(s) >= L else 0.0
    row["roll7"] = s.shift(1).rolling(7, min_periods=1).mean().iloc[-1] if len(s) >= 1 else 0.0
    row["day_index"] = (s.index.max() - s.index.min()).days + 1
    return pd.DataFrame([row])


# ---------- 1) Load & merge ----------
df1 = safe_read(FILE1)
df2 = safe_read(FILE2)
print("Loaded shapes:", df1.shape, df2.shape)

if "salary_local" in df2.columns and "salary_local" not in df1.columns:
    df2 = df2.drop(columns=["salary_local"])

common = [c for c in df1.columns if c in df2.columns]
df = pd.concat([df1[common], df2[common]], ignore_index=True)
print("Merged:", df.shape)

# ---------- 2) Dates & skills ----------
if "posting_date" not in df.columns:
    raise RuntimeError("posting_date missing")
df["posting_date"] = parse_dates_safe(df["posting_date"])
n_invalid = df["posting_date"].isna().sum()
if n_invalid:
    print(f"Dropping {n_invalid} rows with invalid posting_date")
df = df.dropna(subset=["posting_date"]).reset_index(drop=True)
df["posting_date"] = df["posting_date"].dt.normalize()
print("Date range:", df["posting_date"].min(), "→", df["posting_date"].max())

if "required_skills" not in df.columns:
    raise RuntimeError("required_skills missing")
df["required_skills"] = df["required_skills"].astype(str).str.strip()
df["skills_list"] = df["required_skills"].apply(lambda s: [t.strip() for t in s.split(",") if t.strip()])

# ---------- 3) Daily counts ----------
skills_daily = (df[["posting_date","skills_list"]].explode("skills_list")
                .rename(columns={"skills_list":"skill"}).dropna(subset=["skill"]))
daily_counts = skills_daily.groupby(["posting_date","skill"]).size().reset_index(name="count")
full_idx = pd.date_range(daily_counts["posting_date"].min(), daily_counts["posting_date"].max(), freq="D")

top_skills = (daily_counts.groupby("skill")["count"].sum().sort_values(ascending=False).head(TOP_K_SKILLS).index.tolist())
print("Top skills sample:", top_skills[:8])

daily_counts = daily_counts[daily_counts["skill"].isin(top_skills)]
skill_matrix = (daily_counts.pivot(index="posting_date", columns="skill", values="count")
                .reindex(full_idx).fillna(0.0).rename_axis(index="date", columns=None))
print("Skill matrix:", skill_matrix.shape)

# ---------- 4) Split ----------
if skill_matrix.shape[0] <= (TEST_DAYS + 30):
    TEST_DAYS = max(7, skill_matrix.shape[0] // 10)
print("TEST_DAYS:", TEST_DAYS)
train_df = skill_matrix.iloc[:-TEST_DAYS]
test_df  = skill_matrix.iloc[-TEST_DAYS:]

# ---------- 5) Train 1-step & direct (without early stopping) ----------
models_1step = {}
models_direct = {}
metrics_recursive = []
metrics_direct = []

t0 = time.time()
for skill in skill_matrix.columns:
    total = int(skill_matrix[skill].sum())
    if total < MIN_TOTAL_COUNT:
        continue

    # 1-step
    try:
        X_train_1, y_train_1 = make_supervised(train_df[skill], max_lag=MAX_LAG)
    except Exception:
        continue
    if len(y_train_1) < (MAX_LAG + 7):
        continue

    # simple internal split (no early stopping param)
    split_idx = int(len(X_train_1) * 0.9)
    X_tr, X_val = X_train_1.iloc[:split_idx], X_train_1.iloc[split_idx:]
    y_tr, y_val = y_train_1.iloc[:split_idx], y_train_1.iloc[split_idx:]

    m1 = LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        n_estimators=N_ESTIMATORS,
        n_jobs=-1,
        verbosity=-1
    )
    # Fit without early_stopping_rounds to stay compatible
    m1.fit(X_tr, y_tr)
    models_1step[skill] = m1

    # recursive backtest
    history = train_df[skill].copy()
    preds = []
    for target_date, true_val in test_df[skill].items():
        X_hist, _ = make_supervised(history, max_lag=MAX_LAG)
        if X_hist.empty:
            y_hat = 0.0
        else:
            y_hat = float(m1.predict(X_hist.iloc[-1:].values)[0])
        preds.append(y_hat)
        history = pd.concat([history, pd.Series([y_hat], index=[target_date])])
    y_true = test_df[skill].values
    y_pred = np.array(preds)
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.maximum(1e-6, y_true)
    mape = float((np.abs(y_true - y_pred) / denom).mean() * 100.0)
    metrics_recursive.append({"skill": skill, "MAE": mae, "MAPE%": mape, "total": total})

    # direct horizons
    models_direct[skill] = {}
    for h in DIRECT_HORIZONS:
        s = train_df[skill]
        df_sup = pd.DataFrame({"y": s})
        for L in range(1, MAX_LAG+1):
            df_sup[f"lag_{L}"] = s.shift(L)
        df_sup["roll7"] = s.shift(1).rolling(7, min_periods=1).mean()
        df_sup["day_index"] = (s.index - s.index.min()).days
        df_sup[f"y_h{h}"] = s.shift(-h)
        df_sup = df_sup.dropna()
        if df_sup.shape[0] < 30:
            continue
        Xd = df_sup.drop(columns=["y", f"y_h{h}"])
        yd = df_sup[f"y_h{h}"]
        sidx = int(len(Xd)*0.9)
        Xd_tr, Xd_val = Xd.iloc[:sidx], Xd.iloc[sidx:]
        yd_tr, yd_val = yd.iloc[:sidx], yd.iloc[sidx:]
        md = LGBMRegressor(
            objective="regression",
            random_state=RANDOM_STATE,
            learning_rate=LEARNING_RATE,
            num_leaves=NUM_LEAVES,
            min_data_in_leaf=MIN_DATA_IN_LEAF,
            n_estimators=N_ESTIMATORS,
            n_jobs=-1,
            verbosity=-1
        )
        md.fit(Xd_tr, yd_tr)
        models_direct[skill][h] = md
        yhat_d = md.predict(Xd)
        mae_d = float(mean_absolute_error(yd, yhat_d))
        denom_d = np.maximum(1e-6, yd.to_numpy())
        mape_d = float((np.abs(yd - yhat_d) / denom_d).mean() * 100.0)
        metrics_direct.append({"skill": skill, "horizon": h, "MAE": mae_d, "MAPE%": mape_d})

t1 = time.time()
print("Training completed. 1-step models:", len(models_1step), "direct-horizon entries:", sum(len(v) for v in models_direct.values()))
print("Time:", round(t1-t0,1), "s")

# Save metrics CSVs
df_rec = pd.DataFrame(metrics_recursive).sort_values("MAE") if metrics_recursive else pd.DataFrame()
df_dir = pd.DataFrame(metrics_direct).sort_values(["skill","horizon"]) if metrics_direct else pd.DataFrame()
if not df_rec.empty:
    df_rec.to_csv("skill_backtest_recursive_metrics.csv", index=False)
    print("Saved -> skill_backtest_recursive_metrics.csv")
if not df_dir.empty:
    df_dir.to_csv("skill_direct_horizon_metrics.csv", index=False)
    print("Saved -> skill_direct_horizon_metrics.csv")

# ---------- 6) Forecast future (recursive + direct where available) ----------
today = pd.Timestamp.today().normalize()
last_hist = skill_matrix.index.max()
gap = max(0, (today - last_hist).days)
HORIZON = gap + 60
future_dates = pd.date_range(last_hist + timedelta(days=1), periods=HORIZON, freq="D")
forecast_matrix = pd.DataFrame(index=future_dates, columns=skill_matrix.columns, dtype=float)

for skill in skill_matrix.columns:
    history = skill_matrix[skill].copy()
    for i, d in enumerate(future_dates, start=1):
        direct = models_direct.get(skill, {}).get(i, None)
        if direct is not None:
            Xrow = build_direct_row_from_history(history, max_lag=MAX_LAG)
            try:
                yhat = float(direct.predict(Xrow)[0])
            except Exception:
                yhat = 0.0
        else:
            m1 = models_1step.get(skill, None)
            if m1 is None:
                yhat = 0.0
            else:
                X_hist, _ = make_supervised(history, max_lag=MAX_LAG)
                if X_hist.empty:
                    yhat = 0.0
                else:
                    yhat = float(m1.predict(X_hist.iloc[-1:].values)[0])
        forecast_matrix.loc[d, skill] = max(0.0, yhat)
        history = pd.concat([history, pd.Series([yhat], index=[d])])

print("Forecast range:", forecast_matrix.index.min().date(), "→", forecast_matrix.index.max().date())

# ---------- 7) Query helpers ----------
def _target_date(days_ahead):
    return pd.Timestamp.today().normalize() + timedelta(days=days_ahead)

def _value_on_date(skill, dt):
    if dt <= last_hist:
        try:
            return float(skill_matrix.loc[dt, skill])
        except Exception:
            return 0.0
    else:
        try:
            return float(forecast_matrix.loc[dt, skill])
        except Exception:
            return 0.0

def top_k_skills(days_ahead: int, k=5):
    if not (0 <= days_ahead <= 60):
        raise ValueError("days_ahead must be 0..60")
    dt = _target_date(days_ahead)
    scores = {s: _value_on_date(s, dt) for s in skill_matrix.columns}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return dt, ranked


# ---------- 8) Demo & save ----------
try:
    user_days = int(input("Enter days ahead (0..60), default 20: ").strip() or 20)
except Exception:
    user_days = 20

dt, top5 = top_k_skills(user_days, k=5)
print(f"\nTop 5 skills on {dt.date()} (d+{user_days}):")
for s,v in top5:
    print(f"  {s:<20} {v:.2f}")

pd.DataFrame(top5, columns=["skill","predicted_count"]).to_csv(f"top5_day{user_days}_predictions.csv", index=False)
print("Saved ->", f"top5_day{user_days}_predictions.csv")

# ---------- 9) Plots ----------
plt.figure(figsize=(8,4))
labels = [s for s,_ in top5]; vals = [v for _,v in top5]
plt.barh(labels[::-1], vals[::-1], color="tab:orange")
plt.title(f"Top {len(top5)} Predicted Skills on {dt.date()} (d+{user_days})")
plt.xlabel("Predicted postings"); plt.tight_layout(); plt.show()

def plot_history_forecast(skill, days_back=240):
    if skill not in skill_matrix.columns:
        print("not modeled:", skill); return
    hist = skill_matrix[skill]
    cutoff = hist.index.max() - pd.Timedelta(days=days_back)
    hist_plot = hist[hist.index >= cutoff]
    fc = forecast_matrix[skill].dropna()
    plt.figure(figsize=(12,4))
    plt.plot(hist_plot.index, hist_plot.values, label="History")
    if not fc.empty:
        plt.plot(fc.index, fc.values, linestyle="--", label="Forecast")
    plt.title(f"History vs Forecast — {skill}"); plt.xlabel("Date"); plt.ylabel("Postings")
    plt.legend(); plt.tight_layout(); plt.show()

if top5:
    plot_history_forecast(top5[0][0], days_back=240)

# ---------- 10) Report summary ----------
if not df_rec.empty:
    print("\nRecursive backtest summary:")
    print("Skills modeled:", len(df_rec))
    print("Macro MAE:", round(df_rec["MAE"].mean(), 3))
    print("Macro MAPE%:", round(df_rec["MAPE%"].mean(), 2))
    print("\nTop 5 best MAE:")
    print(df_rec.head(5)[["skill","MAE","MAPE%"]])
    print("\nTop 5 worst MAE:")
    print(df_rec.tail(5)[["skill","MAE","MAPE%"]])

if not df_dir.empty:
    print("\nDirect-horizon sample metrics (first rows):")
    print(df_dir.head())

print("Total time (s):", round(time.time() - t_start, 1))