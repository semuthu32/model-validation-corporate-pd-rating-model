################################################################################
#################### DISCRIMINATORY POWER (Train + OoT) ########################
################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
from catboost import CatBoostClassifier

# ---------------------------
# Config / files
# ---------------------------
DATA_DIR = "./DATA"
OUT_DIR = "./Output"
TRAIN_FILE = os.path.join(DATA_DIR, "Train.csv")
OOT_FILE = os.path.join(DATA_DIR, "OoT.csv")
MODEL_OUTPUTS_FILE = os.path.join(OUT_DIR, "Model_Outputs.csv")  # OoT predictions already saved
MODEL_FILE = os.path.join(OUT_DIR, "catboost_classifier")

# Known categorical features from doc
CATEGORICAL_FEATURES = ['X21', 'X43', 'X55']  # update if different

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Load data & model
# ---------------------------
train_df = pd.read_csv(TRAIN_FILE)
oot_df = pd.read_csv(OOT_FILE)
oo_preds = pd.read_csv(MODEL_OUTPUTS_FILE)   # contains ID, TARGET, Pred, p_calib, p_calib_bayes

model = CatBoostClassifier()
model.load_model(MODEL_FILE)

print("\n==== SLIDE 3: DISCRIMINATORY POWER (Train + OoT) ====\n")

# ---------------------------
# Helper functions
# ---------------------------
def compute_metrics(y_true, y_score):
    auc_val = roc_auc_score(y_true, y_score)
    gini = 2 * auc_val - 1
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    ks_thresh = thresholds[np.argmax(tpr - fpr)]
    return {'auc': auc_val, 'gini': gini, 'ks': ks, 'ks_thresh': ks_thresh, 'fpr': fpr, 'tpr': tpr}

def safe_predict(model, df, cat_cols):
    """
    Predict using model.predict_proba on df.
    - Ensures df has exactly the model feature columns in the same order.
    - Converts only the specified categorical columns to string.
    """
    X = df.copy()
    # Convert only the categorical columns that exist in X
    cat_present = [c for c in cat_cols if c in X.columns]
    for c in cat_present:
        X[c] = X[c].astype(str)

    # Ensure we select exactly the features the model expects and in the correct order
    # Some training code removed correlated columns; so model.feature_names_ is the source of truth
    model_feats = list(model.feature_names_)
    missing = [f for f in model_feats if f not in X.columns]
    if missing:
        raise ValueError(f"Missing feature columns required by model: {missing}")

    X_model = X[model_feats]
    # Predict (this internally builds a Pool, but types are safe because only cat_present are str)
    preds = model.predict_proba(X_model)[:, 1]
    return preds

# ---------------------------
# 1) OoT analysis (use saved predictions)
# ---------------------------
print("Running OoT analysis (using saved predictions)...")
oot = oo_preds.copy()  # has Pred, p_calib, p_calib_bayes, TARGET, ID
# Ensure columns exist
for col in ['Pred', 'p_calib', 'p_calib_bayes']:
    if col not in oot.columns:
        raise ValueError(f"Expected column '{col}' in {MODEL_OUTPUTS_FILE}")

# compute metrics for raw and calibrated predictions
metrics = {}
for col in ['Pred', 'p_calib', 'p_calib_bayes']:
    m = compute_metrics(oot['TARGET'].values, oot[col].values)
    metrics[col] = m
    print(f"OoT - {col}: AUC={m['auc']:.4f}, Gini={m['gini']:.4f}, KS={m['ks']:.4f} (KS_thresh={m['ks_thresh']:.6f})")

# Decile table for raw Pred (also can be produced for calib versions if needed)
oot['Decile_Pred'] = pd.qcut(oot['Pred'], q=10, labels=False, duplicates='drop')
decile_table = oot.groupby('Decile_Pred', observed=True).agg(
    Count=('TARGET', 'count'),
    Defaults=('TARGET', 'sum'),
    Default_Rate=('TARGET', 'mean'),
    Score_Min=('Pred', 'min'),
    Score_Max=('Pred', 'max'),
    Score_Mean=('Pred', 'mean')
).round(6)
decile_table.to_csv(os.path.join(OUT_DIR, "Slide3_Deciles_OoT.csv"), index=True)
print("Saved OoT decile table ->", os.path.join(OUT_DIR, "Slide3_Deciles_OoT.csv"))

# ROC plot (compare raw & calib)
plt.figure(figsize=(8,6))
# Model ROC curves
for col, style, color in zip(
    ['Pred', 'p_calib', 'p_calib_bayes'],
    ['-', '--', ':'],
    ['steelblue', 'orange', 'green']
):
    m = metrics[col]
    plt.plot(
        m['fpr'], m['tpr'],
        lw=2.0,
        linestyle=style,
        color=color,
        label=f"{col} (AUC={m['auc']:.4f})"
    )

# Random classifier (diagonal baseline)
plt.plot([0, 1], [0, 1],linestyle='-.',color='gray',linewidth=1.0,label='Random classifier (AUC=0.5)')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Out-of-Time")
plt.legend(loc='lower right')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Slide3_ROC_OoT.png"), dpi=200)
plt.close()
print("Saved ROC ->", os.path.join(OUT_DIR, "Slide3_ROC_OoT.png"))

# KS plot (raw Pred)
m_raw = metrics['Pred']
good = oot[oot['TARGET']==0]['Pred'].sort_values()
bad  = oot[oot['TARGET']==1]['Pred'].sort_values()
good_cum = np.arange(1, len(good)+1)/len(good) if len(good)>0 else np.array([])
bad_cum  = np.arange(1, len(bad)+1)/len(bad) if len(bad)>0 else np.array([])

plt.figure(figsize=(10,6))
if len(good)>0: plt.plot(good, good_cum, label=f"Non-Defaulters (n={len(good)})", lw=2)
if len(bad)>0:  plt.plot(bad,  bad_cum,  label=f"Defaulters (n={len(bad)})", lw=2)
plt.axvline(m_raw['ks_thresh'], color='black', linestyle='--', label=f"KS={m_raw['ks']:.4f}")
plt.xlabel("Predicted PD")
plt.ylabel("Cumulative Probability")
plt.title("KS Plot – OoT")
plt.legend(loc='lower right')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Slide3_KS_OoT.png"), dpi=200)
plt.close()
print("Saved KS ->", os.path.join(OUT_DIR, "Slide3_KS_OoT.png"))

# Score distribution
plt.figure(figsize=(10,5))
plt.hist(good, bins=40, alpha=0.6, label='Non-Defaulters', edgecolor='black')
plt.hist(bad,  bins=30, alpha=0.6, label='Defaulters', edgecolor='black')
plt.xlabel("Predicted PD")
plt.ylabel("Frequency")
plt.title("Score Distribution – OoT")
plt.legend()
plt.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Slide3_ScoreDist_OoT.png"), dpi=200)
plt.close()
print("Saved Score Dist ->", os.path.join(OUT_DIR, "Slide3_ScoreDist_OoT.png"))

# Feature importance (model)
feat_imp = pd.DataFrame({
    'Feature': model.feature_names_,
    'Importance': model.get_feature_importance()
}).sort_values('Importance', ascending=False)
feat_imp.to_csv(os.path.join(OUT_DIR, "Slide3_FeatureImportance.csv"), index=False)
# top15 plot
top15 = feat_imp.head(15)
plt.figure(figsize=(10,6))
sns.barplot(data=top15, x='Importance', y='Feature', color='steelblue')
plt.title("Top 15 Features")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Slide3_FeatureImportance.png"), dpi=200)
plt.close()
print("Saved Feature Importance ->", os.path.join(OUT_DIR, "Slide3_FeatureImportance.png"))



# ---------------------------
# 2) Optional: Safe Train predictions (stability check)
# ---------------------------
print("\nOptional: computing Train metrics (stability check) using saved model")

# Build safe train copy
train_safe = train_df.copy()
# Convert only known categorical columns to string
cat_present = [c for c in CATEGORICAL_FEATURES if c in train_safe.columns]
for c in cat_present:
    train_safe[c] = train_safe[c].astype(str)

# Predict on train using model.predict_proba with proper column order
try:
    train_preds = safe_predict(model, train_safe, cat_present)
    train_metrics = compute_metrics(train_safe['TARGET'].values, train_preds)
    print(
        f"Train metrics: AUC={train_metrics['auc']:.4f}, "
        f"Gini={train_metrics['gini']:.4f}, KS={train_metrics['ks']:.4f}"
    )

    # Save Train deciles
    train_safe['Pred'] = train_preds
    train_safe['Decile_Pred'] = pd.qcut(train_safe['Pred'], 10, labels=False, duplicates='drop')
    train_deciles = train_safe.groupby('Decile_Pred', observed=True).agg(
        Count=('TARGET', 'count'),
        Defaults=('TARGET', 'sum'),
        Default_Rate=('TARGET', 'mean'),
        Score_Min=('Pred', 'min'),
        Score_Max=('Pred', 'max'),
        Score_Mean=('Pred', 'mean')
    ).round(6)
    train_deciles.to_csv(os.path.join(OUT_DIR, "Slide3_Deciles_Train.csv"))
    print("Saved Train deciles ->", os.path.join(OUT_DIR, "Slide3_Deciles_Train.csv"))

    # Combined summary
    summary_df = pd.DataFrame({
        'Dataset': ['Train', 'OoT'],
        'AUC': [train_metrics['auc'], metrics['Pred']['auc']],
        'Gini': [train_metrics['gini'], metrics['Pred']['gini']],
        'KS': [train_metrics['ks'], metrics['Pred']['ks']]
    })
    summary_df.to_csv(os.path.join(OUT_DIR, "Slide3_Train_OoT_Metrics.csv"), index=False)
    print("Saved Train vs OoT summary ->", os.path.join(OUT_DIR, "Slide3_Train_OoT_Metrics.csv"))

except Exception as e:
    print("WARNING: Train prediction failed (safe fallback). Error:")
    print(e)
    # continue without Train metrics

# ---------------------------
# Final OoT summary
# ---------------------------
summary_oot = pd.DataFrame({
    'Metric': ['AUC','Gini','KS','KS Threshold','Defaulters (OoT)','Non-Defaulters (OoT)'],
    'Value': [f"{metrics['Pred']['auc']:.4f}", f"{metrics['Pred']['gini']:.4f}", f"{metrics['Pred']['ks']:.4f}",
              f"{metrics['Pred']['ks_thresh']:.6f}", len(bad), len(good)]
})
summary_oot.to_csv(os.path.join(OUT_DIR, "Slide3_Discriminatory_Summary_OoT.csv"), index=False)
print("\nSaved final OoT summary ->", os.path.join(OUT_DIR, "Slide3_Discriminatory_Summary_OoT.csv"))

# =====================================================
# SIMPLE LEAKAGE CHECK SUMMARY
# =====================================================

print("\nRunning simple leakage checks...")

# 1) Correlation of features with TARGET (numeric only)
num_cols = [c for c in train_df.columns
            if c not in ['ID', 'TARGET'] and train_df[c].dtype != 'object']

corrs = train_df[num_cols].corrwith(train_df['TARGET']).sort_values(ascending=False)
corrs.to_csv(os.path.join(OUT_DIR, "Leakage_CorrToTARGET.csv"))

# Flag potential leakage if |corr| > 0.30
leak_flags_corr = corrs[abs(corrs) > 0.30]

# 2) Missingness difference between default vs non-default
miss_diff = {}
for col in train_df.columns:
    if col in ['ID', 'TARGET']:
        continue
    m0 = train_df[train_df['TARGET']==0][col].isna().mean()
    m1 = train_df[train_df['TARGET']==1][col].isna().mean()
    miss_diff[col] = abs(m1 - m0)

miss_diff = pd.Series(miss_diff).sort_values(ascending=False)
miss_diff.to_csv(os.path.join(OUT_DIR, "Leakage_MissingnessDiff.csv"))

# Flag if missingness difference > 0.20
leak_flags_miss = miss_diff[miss_diff > 0.20]

# 3) Save a simple summary file
summary = pd.DataFrame({
    'CorrToTARGET': corrs,
    'MissingnessDiff': miss_diff,
})
summary.to_csv(os.path.join(OUT_DIR, "Leakage_Summary.csv"))

print("Leakage Checks Saved →", os.path.join(OUT_DIR, "Leakage_Summary.csv"))
print("High-correlation flags:", leak_flags_corr.index.tolist())
print("Missingness-diff flags:", leak_flags_miss.index.tolist())

print("\n==== Completed Slide 3 outputs ====\n")