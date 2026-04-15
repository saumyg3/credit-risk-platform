# =============================================================================
# Credit Risk Platform — Model Training Pipeline
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data(path: str = "data/featured.parquet"):
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in ["default", "grade", "purpose", "loan_status"]]
    X = df[feature_cols]
    y = df["default"]
    return X, y, df, feature_cols


def train_all_models(X_train, y_train):
    models = {}

    # ── Logistic Regression (baseline) ────────────────────────────────
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr
    print("  Done!")

    # ── Random Forest ──────────────────────────────────────────────────
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=50,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    print("  Done!")

    # ── XGBoost ────────────────────────────────────────────────────────
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=4,  # handle class imbalance
        random_state=42, n_jobs=-1, eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb
    print("  Done!")

    # ── LightGBM ───────────────────────────────────────────────────────
    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=4,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    models["LightGBM"] = lgbm
    print("  Done!")

    return models


def evaluate_models(models, X_test, y_test):
    print("\n── Model Evaluation Results ──")
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "AUC-ROC": auc,
            "Avg Precision": ap,
            "Precision (Default)": report["1"]["precision"],
            "Recall (Default)": report["1"]["recall"],
            "F1 (Default)": report["1"]["f1-score"],
        }
        print(f"\n{name}:")
        print(f"  AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f}")
        print(f"  Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f} | F1: {report['1']['f1-score']:.4f}")

    return results


def plot_model_comparison(results):
    metrics = ["AUC-ROC", "Avg Precision", "F1 (Default)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Credit Risk Model Comparison", fontsize=14, fontweight="bold")

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    for idx, metric in enumerate(metrics):
        names = list(results.keys())
        values = [results[n][metric] for n in names]
        bars = axes[idx].bar(names, values, color=colors)
        axes[idx].set_title(metric)
        axes[idx].set_ylim(0, 1)
        axes[idx].tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/model_comparison.png")


def compute_shap(model, X_train, X_test, feature_cols, model_name="XGBoost"):
    print(f"\nComputing SHAP values for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:2000])

    # Global feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"SHAP Feature Importance — {model_name}", fontsize=13, fontweight="bold")

    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_test[:2000], feature_names=feature_cols,
                      show=False, plot_type="bar")
    axes[0].set_title("Mean |SHAP| — Global Importance")

    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_test[:2000], feature_names=feature_cols, show=False)
    axes[1].set_title("SHAP Value Distribution")

    plt.tight_layout()
    plt.savefig("results/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/shap_importance.png")

    return explainer, shap_values


def fairness_analysis(df, y_pred_prob, threshold=0.5):
    print("\n── Fairness Analysis ──")
    df = df.copy()
    df["predicted_default"] = (y_pred_prob >= threshold).astype(int)
    df["default_prob"] = y_pred_prob

    # ── By Loan Grade ──────────────────────────────────────────────────
    grade_analysis = df.groupby("grade").agg(
        actual_default_rate=("default", "mean"),
        predicted_default_rate=("predicted_default", "mean"),
        avg_risk_score=("default_prob", "mean"),
        count=("default", "count")
    ).round(4)
    print("\nDefault Rate by Loan Grade:")
    print(grade_analysis.to_string())

    # ── By Purpose ────────────────────────────────────────────────────
    purpose_analysis = df.groupby("purpose").agg(
        actual_default_rate=("default", "mean"),
        predicted_default_rate=("predicted_default", "mean"),
        avg_risk_score=("default_prob", "mean"),
        count=("default", "count")
    ).sort_values("avg_risk_score", ascending=False).round(4)
    print("\nDefault Rate by Loan Purpose:")
    print(purpose_analysis.to_string())

    # ── Income Band Analysis ───────────────────────────────────────────
    df["income_band"] = pd.cut(df["annual_inc"],
        bins=[0, 40000, 60000, 80000, 100000, float("inf")],
        labels=["<40k", "40-60k", "60-80k", "80-100k", "100k+"]
    )
    income_analysis = df.groupby("income_band", observed=True).agg(
        actual_default_rate=("default", "mean"),
        predicted_default_rate=("predicted_default", "mean"),
        approval_rate=("predicted_default", lambda x: 1 - x.mean()),
        count=("default", "count")
    ).round(4)
    print("\nDisparate Impact by Income Band:")
    print(income_analysis.to_string())

    # ── Disparate Impact Ratio ────────────────────────────────────────
    approval_rates = income_analysis["approval_rate"]
    max_rate = approval_rates.max()
    min_rate = approval_rates.min()
    di_ratio = min_rate / max_rate
    print(f"\nDisparate Impact Ratio (min/max approval): {di_ratio:.4f}")
    if di_ratio < 0.8:
        print("  ⚠️  Disparate impact detected (ratio < 0.8 — CFPB threshold)")
    else:
        print("  ✅  No significant disparate impact detected")

    # ── Plot fairness ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fairness Analysis — Credit Risk Model", fontsize=13, fontweight="bold")

    grade_analysis["actual_default_rate"].plot(kind="bar", ax=axes[0],
        color="#e74c3c", alpha=0.7, label="Actual")
    grade_analysis["predicted_default_rate"].plot(kind="bar", ax=axes[0],
        color="#3498db", alpha=0.7, label="Predicted")
    axes[0].set_title("Default Rate by Grade")
    axes[0].set_xlabel("Grade")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=0)

    income_analysis["approval_rate"].plot(kind="bar", ax=axes[1], color="#2ecc71")
    axes[1].set_title("Approval Rate by Income Band")
    axes[1].set_xlabel("Income Band")
    axes[1].set_ylabel("Approval Rate")
    axes[1].axhline(y=0.8, color="red", linestyle="--", label="0.8 CFPB threshold")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig("results/fairness_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/fairness_analysis.png")

    return grade_analysis, purpose_analysis, income_analysis


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Load data
    X, y, df, feature_cols = load_data()
    print(f"Dataset: {X.shape}, Default rate: {y.mean()*100:.1f}%")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    df_test = df.iloc[y_test.index]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

    # Train all models
    models = train_all_models(X_train_scaled, y_train)

    # Evaluate
    results = evaluate_models(models, X_test_scaled, y_test)

    # Plot comparison
    plot_model_comparison(results)

    # SHAP on best model (XGBoost)
    explainer, shap_values = compute_shap(
        models["XGBoost"], X_train_scaled, X_test_scaled, feature_cols
    )

    # Fairness analysis
    xgb_probs = models["XGBoost"].predict_proba(X_test_scaled)[:, 1]
    df_test = df.iloc[X_test.index].copy()
    df_test["default"] = y_test.values
    grade_analysis, purpose_analysis, income_analysis = fairness_analysis(df_test, xgb_probs)

    # Save models
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/xgboost.pkl", "wb") as f:
        pickle.dump(models["XGBoost"], f)
    with open("models/all_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("models/explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    print("\n✅ All models saved to models/")
    print("✅ All plots saved to results/")