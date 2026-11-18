# metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

def compute_all_metrics(df_candidates, y_pred_proba):
    """
    df_candidates: DataFrame containing columns ['trip_id', 'obs_idx', 'is_match']
    y_pred_proba: 1D array of predicted probabilities of is_match == 1
    """
    df = df_candidates[['trip_id', 'obs_idx', 'is_match']].copy()
    
    # Fill any NaNs in target labels and convert to int
    df['is_match'] = df['is_match'].fillna(0).astype(int)
    df['prob'] = y_pred_proba
    
    y_true = df['is_match'].values
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Standard candidate-level metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Clip prob to prevent log_loss infinity
    eps = 1e-15
    prob_clipped = np.clip(y_pred_proba, eps, 1 - eps)
    loss = log_loss(y_true, prob_clipped)

    # Observation-level Top-1 Accuracy:
    # Pick the candidate with the highest predicted score for each GPS point
    best_cands = df.sort_values(['trip_id', 'obs_idx', 'prob'], ascending=[True, True, False]) \
                   .groupby(['trip_id', 'obs_idx']).head(1)
    
    top1_acc = best_cands['is_match'].mean()

    return {
        "Macro_F1": macro_f1,
        "Top1_Accuracy": top1_acc,
        "Precision": precision,
        "Recall": recall,
        "Log_Loss": loss
    }