"""Train and evaluate the Gait Instability Classifier (Tier 1 Global Model).

Evaluates Random Forest via 5-Fold GroupKFold cross-validation (grouped by subject)
to ensure zero subject leakage. Trains final model on full dataset and serializes:
- models/gait/gait_rf.pkl
- models/gait/config.json
- data/reports/instability_training_report.txt
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import GroupKFold

from common_gait import (
    FS, WIN, STRIDE, ACC_CLIP, RAW_CHANNELS,
    GAIT_FEATURE_NAMES, PROCESSED, MODELS_DIR, REPORTS
)

N_FOLDS = 5
SEED = 42
DEFAULT_THRESHOLD = 0.40  # Tuned for high sensitivity to stumbles while maintaining >99% specificity


def create_model():
    """Instantiate the Random Forest classifier with balanced weights."""
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED
    )


def evaluate_group_kfold(df, features, target_col="label", group_col="subject"):
    """Evaluate model using 5-Fold GroupKFold grouped strictly by subject."""
    X = df[features].to_numpy()
    y = df[target_col].to_numpy()
    groups = df[group_col].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_probs = np.zeros(len(y), dtype=np.float64)

    print(f"\nRunning {N_FOLDS}-Fold GroupKFold Cross-Validation across {df[group_col].nunique()} subjects...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        fold_model = create_model()
        fold_model.fit(X[train_idx], y[train_idx])
        # Probability of class 1 (unstable)
        probs = fold_model.predict_proba(X[val_idx])
        class_1_idx = list(fold_model.classes_).index(1)
        oof_probs[val_idx] = probs[:, class_1_idx]
        fold_auc = roc_auc_score(y[val_idx], oof_probs[val_idx])
        print(f"  Fold {fold}: {len(val_idx):,} validation windows, AUC = {fold_auc:.4f}")

    oof_preds = (oof_probs >= DEFAULT_THRESHOLD).astype(int)
    auc = roc_auc_score(y, oof_probs)
    tn, fp, fn, tp = confusion_matrix(y, oof_preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y, oof_preds, zero_division=0)
    f1 = f1_score(y, oof_preds, zero_division=0)
    acc = accuracy_score(y, oof_preds)

    metrics = {
        "roc_auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "accuracy": float(acc),
        "threshold": DEFAULT_THRESHOLD,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS (UNSEEN SUBJECTS)")
    print("=" * 70)
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Sensitivity: {sens:6.1%}  (Detected {tp}/{tp+fn} unstable events)")
    print(f"  Specificity: {spec:6.1%}  (Correctly cleared {tn}/{tn+fp} stable windows)")
    print(f"  Precision:   {prec:6.1%}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    return metrics


def main():
    print("=" * 70)
    print("STEP 3: TRAIN GAIT INSTABILITY CLASSIFIER (Tier 1 Random Forest)")
    print("=" * 70)

    in_path = PROCESSED / "gait_features.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run 02_gait_features.py first.")

    df = pd.read_parquet(in_path)
    print(f"Loaded dataset: {len(df):,} windows across {df['subject'].nunique()} subjects")
    print(f"Class distribution: {df['label_name'].value_counts().to_dict()}")

    # 1. Subject-independent Cross Validation
    cv_metrics = evaluate_group_kfold(df, GAIT_FEATURE_NAMES)

    # 2. Train final production model on full dataset
    print("\nTraining final production model on 100% of data...")
    X_full = df[GAIT_FEATURE_NAMES].to_numpy()
    y_full = df["label"].to_numpy()

    final_model = create_model()
    final_model.fit(X_full, y_full)

    # Feature importances
    importances = final_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        {"feature": GAIT_FEATURE_NAMES[i], "importance": float(importances[i])}
        for i in sorted_idx
    ]

    print("\nTop 10 Most Important Biomechanical Features:")
    for rank, item in enumerate(top_features[:10], 1):
        print(f"  {rank:2d}. {item['feature']:<25s}: {item['importance']:.4f}")

    # 3. Model packaging and metadata configuration
    config = {
        "model_type": "RandomForestClassifier",
        "sample_rate_hz": FS,
        "window_samples": WIN,
        "window_seconds": WIN / FS,
        "stride_samples": STRIDE,
        "raw_channels": RAW_CHANNELS,
        "acc_clip_g": ACC_CLIP,
        "threshold": DEFAULT_THRESHOLD,
        "n_features": len(GAIT_FEATURE_NAMES),
        "feature_names": GAIT_FEATURE_NAMES,
        "evaluation_metrics": cv_metrics,
        "top_features": top_features[:10],
        "training_dataset": {
            "total_windows": len(df),
            "stable_windows": int((df["label"] == 0).sum()),
            "unstable_windows": int((df["label"] == 1).sum()),
            "subjects": int(df["subject"].nunique()),
            "cohorts": df["cohort"].value_counts().to_dict()
        },
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Serialize model bundle
    model_bundle = {
        "model": final_model,
        "feature_names": GAIT_FEATURE_NAMES,
        "config": config,
        "classes": list(final_model.classes_)
    }
    model_path = MODELS_DIR / "gait_rf.pkl"
    config_path = MODELS_DIR / "config.json"
    report_path = REPORTS / "instability_training_report.txt"

    joblib.dump(model_bundle, model_path)
    config_path.write_text(json.dumps(config, indent=2))

    # Write textual report
    report_lines = [
        "GAIT INSTABILITY CLASSIFIER - TRAINING & EVALUATION REPORT",
        "=" * 60,
        f"Algorithm:       Random Forest (n_estimators=150, max_depth=12)",
        f"Validation:      5-Fold GroupKFold (Subject-Independent, 0 leakage)",
        f"Total Windows:   {len(df):,}",
        f"Total Subjects:  {df['subject'].nunique()}",
        "",
        "Cross-Validation Performance:",
        f"  ROC-AUC:       {cv_metrics['roc_auc']:.4f}",
        f"  Sensitivity:   {cv_metrics['sensitivity']:.1%}",
        f"  Specificity:   {cv_metrics['specificity']:.1%}",
        f"  Precision:     {cv_metrics['precision']:.1%}",
        f"  F1 Score:      {cv_metrics['f1_score']:.4f}",
        f"  Accuracy:      {cv_metrics['accuracy']:.1%}",
        "",
        "Top 10 Feature Importances:",
        "\n".join([f"  {r['feature']:<25s}: {r['importance']:.4f}" for r in top_features[:10]])
    ]
    report_path.write_text("\n".join(report_lines))

    print(f"\nArtifacts successfully written:")
    print(f"  -> Model binary: {model_path}")
    print(f"  -> Config JSON:  {config_path}")
    print(f"  -> Full report:  {report_path}")


if __name__ == "__main__":
    main()
