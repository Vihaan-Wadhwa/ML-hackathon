# train_models.py
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from metrics import compute_all_metrics

# 1. Load Data
data_path = Path("data") / "train_candidates.parquet"
print(f"Loading real data from {data_path}...")
df_train = pd.read_parquet(data_path)

target_col = 'is_match'
group_col = 'trip_id'

# Clean target and features
df_train[target_col] = df_train[target_col].fillna(0).astype(int)

feature_cols = [
    'cand_rank', 'perp_dist_m', 'bearing_diff_deg', 'speed_mps', 
    'dt_s', 'gap_flag', 'dist_to_start_m', 'dist_to_end_m', 
    'frac_along', 'seg_len_m', 'n_candidates', 'has_heading'
]
# If any feature column is missing, keep only available ones
feature_cols = [c for c in feature_cols if c in df_train.columns]
df_train[feature_cols] = df_train[feature_cols].fillna(0)

print(f"Dataset Shape: {df_train.shape} across {df_train[group_col].nunique()} trips.")
print(f"Selected {len(feature_cols)} features: {feature_cols}")

# Extract NumPy arrays for fast modeling
X = df_train[feature_cols].values
y = df_train[target_col].values
groups = df_train[group_col].values

# 2. Define 5 Distinct Algorithm Families
models = {
    "Logistic_Regression (Linear)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision_Tree (Tree)": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random_Forest (Ensemble)": RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1),
    "HistGradientBoosting (Boosting)": HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.08, random_state=42),
    "MLP_NeuralNet (Neural Network)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42))
}

# 3. GroupKFold Cross Validation (5 Folds)
gkf = GroupKFold(n_splits=5)
results = []

print("\nStarting 5-Model Benchmark on Real Data with GroupKFold...")
for name, model in models.items():
    start_time = time.time()
    cv_top1, cv_f1, cv_loss = [], [], []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        # Correct NumPy indexing: no .iloc!
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        val_df = df_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)[:, 1]
        
        m = compute_all_metrics(val_df, probs)
        cv_top1.append(m["Top1_Accuracy"])
        cv_f1.append(m["Macro_F1"])
        cv_loss.append(m["Log_Loss"])

    total_time = time.time() - start_time
    results.append({
        "Model": name,
        "CV_Top1_Accuracy": np.mean(cv_top1),
        "CV_Macro_F1": np.mean(cv_f1),
        "CV_Log_Loss": np.mean(cv_loss),
        "Fit_Inference_Time_Sec": round(total_time, 2)
    })
    print(f"-> {name} complete: Top-1 = {np.mean(cv_top1):.4f}, Macro F1 = {np.mean(cv_f1):.4f}, Time = {round(total_time, 2)}s")

# 4. Save results to CSV for the report
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
summary_df = pd.DataFrame(results)
summary_df.to_csv(results_dir / "model_comparison.csv", index=False)

print("\n=== Model Benchmark Complete ===")
print(summary_df.to_string(index=False))