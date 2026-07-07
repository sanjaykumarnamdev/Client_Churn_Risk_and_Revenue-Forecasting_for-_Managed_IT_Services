"""
Train a churn classifier on the client-level dataset:
- median imputation (+ missingness flag) for the 12%-missing financial field
- one-hot encode categoricals
- class-weighted XGBoost (handles the 18/82 imbalance)
- Stratified k-fold CV (small-N, so avoid a single fragile holdout split)
- SHAP for top churn drivers
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    precision_recall_curve, f1_score
)
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("./client_modeling_dataset.csv")

# ---- missingness flag + median imputation ----
df["revenue_missing_flag"] = df["monthly_revenue"].isna().astype(int)
df["monthly_revenue"] = df["monthly_revenue"].fillna(df["monthly_revenue"].median())

# ---- encode categoricals ----
df_model = pd.get_dummies(df, columns=["industry", "size_tier"], drop_first=True)

feature_cols = [c for c in df_model.columns if c not in ("client_id", "will_churn_3mo")]
X = df_model[feature_cols]
y = df_model["will_churn_3mo"]

print(f"N clients: {len(df)}, churn rate: {y.mean():.3f}, features: {len(feature_cols)}")

# ---- class weight for imbalance ----
neg, pos = (y == 0).sum(), (y == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    min_child_weight=3,
    reg_lambda=2.0,
    eval_metric="aucpr",
    random_state=42,
)

# ---- Stratified 5-fold CV (appropriate given small N=~200) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

roc_auc = roc_auc_score(y, oof_proba)
pr_auc = average_precision_score(y, oof_proba)
print(f"\nOut-of-fold ROC-AUC: {roc_auc:.3f}")
print(f"Out-of-fold PR-AUC : {pr_auc:.3f}  (baseline/random = {y.mean():.3f})")

# pick threshold maximizing F1 on OOF predictions
prec, rec, thresh = precision_recall_curve(y, oof_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.nanargmax(f1s[:-1])
best_thresh = thresh[best_idx]
print(f"Best-F1 threshold: {best_thresh:.3f} -> Precision {prec[best_idx]:.2f}, Recall {rec[best_idx]:.2f}, F1 {f1s[best_idx]:.2f}")

y_pred = (oof_proba >= best_thresh).astype(int)
print("\nClassification report (OOF, chosen threshold):")
print(classification_report(y, y_pred, target_names=["Retained", "Churn"]))

# ---- fit final model on all data for SHAP + scoring ----
final_model = model.fit(X, y)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap}) \
    .sort_values("mean_abs_shap", ascending=False)
print("\nTop churn drivers (SHAP mean |value|):")
print(importance.head(10).to_string(index=False))

importance.to_csv("./shap_feature_importance.csv", index=False)

# ---- plots ----
plt.figure(figsize=(7, 5))
top10 = importance.head(10).iloc[::-1]
plt.barh(top10.feature, top10.mean_abs_shap, color="#4C6EF5")
plt.xlabel("Mean |SHAP value| (impact on churn probability)")
plt.title("Top 10 Churn Drivers")
plt.tight_layout()
plt.savefig("./top_churn_drivers.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, color="#E8590C", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve (PR-AUC = {pr_auc:.2f})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./pr_curve.png", dpi=150)
plt.close()

# ---- save client-level risk scores for the retention team ----
df_scored = df.copy()
df_scored["churn_probability"] = final_model.predict_proba(X)[:, 1]
df_scored["risk_tier"] = pd.cut(
    df_scored["churn_probability"], bins=[-0.01, 0.2, 0.5, 1.01],
    labels=["Green (Low)", "Amber (Medium)", "Red (High)"]
)
df_scored = df_scored.sort_values("churn_probability", ascending=False)
df_scored.to_csv("./client_risk_scores.csv", index=False)

print("\nRisk tier counts:")
print(df_scored.risk_tier.value_counts())
print("\nSaved: shap_feature_importance.csv, top_churn_drivers.png, pr_curve.png, client_risk_scores.csv")

# save metrics summary for reporting
with open("./model_metrics.txt", "w") as f:
    f.write(f"N clients: {len(df)}\n")
    f.write(f"Churn rate: {y.mean():.3f}\n")
    f.write(f"OOF ROC-AUC: {roc_auc:.3f}\n")
    f.write(f"OOF PR-AUC: {pr_auc:.3f}\n")
    f.write(f"Chosen threshold: {best_thresh:.3f}\n")
    f.write(f"Precision at threshold: {prec[best_idx]:.3f}\n")
    f.write(f"Recall at threshold: {rec[best_idx]:.3f}\n")
    f.write(f"F1 at threshold: {f1s[best_idx]:.3f}\n")
