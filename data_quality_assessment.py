################################################################################
############## Data quality assessment / Feature Stability #####################
################################################################################




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Config / Inputs
# ----------------------------
TRAIN_PATH = "./DATA/Train.csv"
OOT_PATH   = "./DATA/OoT.csv"
OUTDIR     = "Output"
ID_COL     = "ID"
TARGET_COL = "TARGET"
CATEGORICAL = ["X21", "X43", "X55"]   # provided
# ----------------------------
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# Load
# ----------------------------
train = pd.read_csv(TRAIN_PATH)
oot   = pd.read_csv(OOT_PATH)

# Basic sanity
assert ID_COL in train.columns and TARGET_COL in train.columns, "ID or TARGET missing in Train"
assert ID_COL in oot.columns and TARGET_COL in oot.columns, "ID or TARGET missing in OoT"

# Define features (exclude ID & TARGET)
all_features = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
numeric_features = [f for f in all_features if f not in CATEGORICAL]
categorical_features = [f for f in CATEGORICAL if f in all_features]

# Coerce numeric features to numeric (safe) to avoid PSI errors from strings
for df in (train, oot):
    for f in numeric_features:
        df[f] = pd.to_numeric(df[f], errors="coerce")

# ----------------------------
# Helper: PSI (robust)
# ----------------------------
def psi_numeric(train_col, oot_col, buckets=10):
    """PSI using train quantile bins. Returns np.nan if not computable."""
    tr = train_col.dropna()
    ot = oot_col.dropna()
    if len(tr) < 50 or tr.nunique() < 3 or ot.nunique() < 1:
        return np.nan
    qs = np.linspace(0, 1, buckets + 1)
    try:
        bins = np.unique(np.quantile(tr, qs))
    except Exception:
        return np.nan
    if len(bins) < 3:
        return np.nan
    tr_counts, _ = np.histogram(tr, bins=bins)
    ot_counts, _ = np.histogram(ot, bins=bins)
    tr_pct = tr_counts / (tr_counts.sum() + 1e-10)
    ot_pct = ot_counts / (ot_counts.sum() + 1e-10)
    eps = 1e-8
    return np.sum((tr_pct - ot_pct) * np.log((tr_pct + eps) / (ot_pct + eps)))

def psi_categorical(train_col, oot_col):
    tr = train_col.fillna("MISSING").astype(str)
    ot = oot_col.fillna("MISSING").astype(str)
    cats = sorted(set(tr.unique()).union(set(ot.unique())))
    tr_dist = tr.value_counts(normalize=True).reindex(cats, fill_value=0)
    ot_dist = ot.value_counts(normalize=True).reindex(cats, fill_value=0)
    eps = 1e-8
    return np.sum((tr_dist - ot_dist) * np.log((tr_dist + eps) / (ot_dist + eps)))

# ----------------------------
# Missingness
# ----------------------------
def missingness_table(df):
    m = df[all_features].isna().sum().to_frame("MissingCount")
    m["MissingPct"] = 100 * m["MissingCount"] / df.shape[0]
    return m.sort_values("MissingPct", ascending=False)

miss_train = missingness_table(train)
miss_oot   = missingness_table(oot)
miss_df = pd.concat([miss_train["MissingPct"], miss_oot["MissingPct"]], axis=1)
miss_df.columns = ["Train_MissingPct", "OoT_MissingPct"]
miss_df.to_csv(os.path.join(OUTDIR, "Missingness_Train_OoT.csv"))

# Plot 1: Missingness top 15
topN = 15
plot_miss = miss_df.head(topN)
plt.figure(figsize=(11,5))
plot_miss.plot.bar(rot=45, figsize=(11,5))
plt.title(f"Top {topN} Features by Missingness (Train vs OoT)")
plt.ylabel("Missing %")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Plot_Missingness_Top15.png"), dpi=300)
plt.close()

# ----------------------------
# Numeric: mean/std/deltas + PSI
# ----------------------------
num_stats = []
psi_num = {}

for f in numeric_features:
    t_mean = train[f].mean(skipna=True)
    o_mean = oot[f].mean(skipna=True)
    t_std  = train[f].std(skipna=True)
    o_std  = oot[f].std(skipna=True)

    num_stats.append({
        "Feature":    f,
        "Train_Mean": t_mean,
        "OoT_Mean":   o_mean,
        "Delta_Mean": o_mean - t_mean,
        "Train_Std":  t_std,
        "OoT_Std":    o_std,
        "Delta_Std":  o_std - t_std
    })

    psi_num[f] = psi_numeric(train[f], oot[f])

num_stats_df = (
    pd.DataFrame(num_stats)
      .sort_values("Delta_Mean", key=lambda col: col.abs(), ascending=False)
)
num_stats_df.to_csv(os.path.join(OUTDIR, "Numeric_Stability.csv"), index=False)

psi_num_df = (
    pd.Series(psi_num, name="PSI")
      .sort_values(ascending=False)
      .to_frame()
)
psi_num_df.to_csv(os.path.join(OUTDIR, "Numeric_PSI.csv"), index=False)

# ----------------------------
# Overall numeric summary (for slides)
# ----------------------------
overall_summary = {
    "Median_Train_Mean": num_stats_df["Train_Mean"].median(),
    "Median_OoT_Mean":   num_stats_df["OoT_Mean"].median(),
    "Median_Train_Std":  num_stats_df["Train_Std"].median(),
    "Median_OoT_Std":    num_stats_df["OoT_Std"].median(),
}

overall_summary_df = pd.DataFrame([overall_summary])
overall_summary_df.to_csv(os.path.join(OUTDIR, "Numeric_Overall_Summary.csv"), index=False)


# Plot 2: PSI top 20 (numeric)
plt.figure(figsize=(10,6))
psi_num_df.head(20).plot.bar(legend=False)
plt.title("Top 20 Numeric Features by PSI (Train vs OoT)")
plt.ylabel("PSI")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Plot_PSI_Top20.png"), dpi=300)
plt.close()


# ----------------------------
# Categorical: frequency tables + PSI
# ----------------------------
cat_tables = {}
cat_psi = {}
for c in categorical_features:
    tr_freq = train[c].fillna("MISSING").astype(str).value_counts(normalize=True)
    ot_freq = oot[c].fillna("MISSING").astype(str).value_counts(normalize=True)
    dfc = pd.concat([tr_freq, ot_freq], axis=1).fillna(0)
    dfc.columns = ["Train_Pct", "OoT_Pct"]
    dfc["Delta"] = dfc["OoT_Pct"] - dfc["Train_Pct"]
    cat_tables[c] = dfc
    dfc.to_csv(os.path.join(OUTDIR, f"Categorical_Stability_{c}.csv"))
    cat_psi[c] = psi_categorical(train[c], oot[c])

pd.Series(cat_psi, name="PSI").to_frame().to_csv(os.path.join(OUTDIR, "Categorical_PSI.csv"))

# ----------------------------
# Console summary (concise)
# ----------------------------
print("\n=== DATA QUALITY & FEATURE STABILITY SUMMARY ===\n")
print(f"Train rows: {len(train):,} | OoT rows: {len(oot):,}")
print("\nTop missing features (Train):")
print(miss_df.head(10).to_string())

print("\nTop numeric PSI (Train vs OoT):")
print(psi_num_df.head(10).to_string())
print("\nNumeric features (median across all features):")
print(overall_summary_df)

# Top numeric features by absolute mean shift
top_mean_shift = num_stats_df.sort_values("Delta_Mean", key=lambda col: col.abs(), ascending=False).head(10)
top_mean_shift.to_csv(os.path.join(OUTDIR, "Numeric_Largest_MeanShift.csv"), index=False)

print("\nTop numeric features by mean shift (Train vs OoT):")
print(top_mean_shift.to_string(index=False))

print("\nCategorical PSI:")
for k,v in cat_psi.items():
    print(f"  {k}: PSI = {v:.6f}")


# ------------------------------
# Default Rates
# ------------------------------
train_default_rate = train['TARGET'].mean()
oot_default_rate = oot['TARGET'].mean()
print(f"\nDefault rates - Train: {train_default_rate:.2%}, OoT: {oot_default_rate:.2%}")

print(f"\nPlots and CSVs saved to '{OUTDIR}/' directory.")

