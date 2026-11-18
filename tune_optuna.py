# tune_optuna.py
from pathlib import Path
import optuna
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from metrics import compute_all_metrics

# 1. Load Datasets
print("Loading datasets for Bayesian Hyperparameter Tuning...")
df_train = pd.read_parquet(Path("data") / "train_candidates.parquet")
df_test = pd.read_parquet(Path("data") / "student_test_candidates.parquet")

target_col = 'is_match'
feature_cols = [
    'cand_rank', 'perp_dist_m', 'bearing_diff_deg', 'speed_mps', 
    'dt_s', 'gap_flag', 'dist_to_start_m', 'dist_to_end_m', 
    'frac_along', 'seg_len_m', 'n_candidates', 'has_heading'
]
feature_cols = [c for c in feature_cols if c in df_train.columns]

# Clean NaNs
df_train[target_col] = df_train[target_col].fillna(0).astype(int)
df_train[feature_cols] = df_train[feature_cols].fillna(0)

df_test[target_col] = df_test[target_col].fillna(0).astype(int)
df_test[feature_cols] = df_test[feature_cols].fillna(0)

X = df_train[feature_cols].values
y = df_train[target_col].values
groups = df_train['trip_id'].values

gkf = GroupKFold(n_splits=5)

# 2. Objective Function for Optuna (Bayesian Optimization)
def objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 60, 200),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 15, 60),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 5.0, log=True),
        'random_state': 42
    }

    f1_scores = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        model = HistGradientBoostingClassifier(**params)
        # Correct NumPy indexing: no .iloc!
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]
        m = compute_all_metrics(df_train.iloc[val_idx], probs)
        f1_scores.append(m['Macro_F1'])

    return np.mean(f1_scores)

print("Running Bayesian Hyperparameter Optimization on Real Data (30 trials)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\nBest Hyperparameters Found:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# 3. Refit Best Model on all 80 training trips
print("\nRefitting best model on all training data...")
best_model = HistGradientBoostingClassifier(**study.best_params, random_state=42)
best_model.fit(X, y)

# 4. Evaluate on Student-Testing Set (Defining X_test)
X_test = df_test[feature_cols].values
test_probs = best_model.predict_proba(X_test)[:, 1]
test_metrics = compute_all_metrics(df_test, test_probs)

print("\nFinal Tuned Model Performance on Student-Test:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

# 5. Save Model and Trial Logs
models_dir = Path("models")
results_dir = Path("results")
models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, models_dir / "final_model.joblib", protocol=4)
study.trials_dataframe().to_csv(results_dir / "bayesian_trials.csv", index=False)

print("\nArtifact saved successfully to models/final_model.joblib!")