"""Per-User Gait Personalization (Tier 2 Unsupervised Anomaly Detection).

Fits an individualized baseline (Isolation Forest + Feature Scaler + Baseline Statistics)
on each subject's normal walking sessions. Alerts trigger when walking deviates
significantly from the user's own established norm.

Outputs:
- models/gait/baselines/<subject>.joblib
- models/gait/baselines/default_elderly.joblib
- models/gait/baselines/default_young.joblib
- data/reports/personalization_report.txt
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from common_gait import (
    GAIT_FEATURE_NAMES, PROCESSED, BASELINES_DIR, REPORTS,
    PersonalGaitBaseline
)


def main():
    print("=" * 70)
    print("STEP 4: PER-USER PERSONALIZATION & BASELINE MODELING")
    print("=" * 70)

    in_path = PROCESSED / "gait_features.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run 02_gait_features.py first.")

    df = pd.read_parquet(in_path)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    subjects = sorted(df["subject"].unique())
    print(f"Building personal baselines for {len(subjects)} subjects...")

    report_lines = [
        "PER-USER PERSONALIZATION EVALUATION REPORT",
        "=" * 60,
        f"{'Subject':10s} {'Cohort':8s} {'Norm Wins':>9s} {'Base Cadence':>13s} {'Base Reg':>9s} {'Stumbles':>9s} {'Detected':>9s}",
        "-" * 70,
    ]

    total_stumbles = 0
    detected_stumbles = 0

    for subj in subjects:
        sub_df = df[df["subject"] == subj]
        norm_windows = sub_df[sub_df["label"] == 0]
        stumble_windows = sub_df[sub_df["label"] == 1]
        cohort = sub_df["cohort"].iloc[0]

        if len(norm_windows) < 30:
            continue

        baseline = PersonalGaitBaseline(user_id=subj)
        baseline.fit(norm_windows)

        # Test anomaly detection on this user's stumble windows
        n_stumble = len(stumble_windows)
        n_detected = 0
        if n_stumble > 0:
            for _, row in stumble_windows.iterrows():
                res = baseline.score_window(row)
                if res["is_anomalous"] or res["anomaly_score"] >= 0.50:
                    n_detected += 1
            total_stumbles += n_stumble
            detected_stumbles += n_detected

        base_cad = baseline.baseline_stats["cadence_fft_hz"]["mean"]
        base_reg = baseline.baseline_stats["step_regularity"]["mean"]

        line = (
            f"{subj:10s} {cohort:8s} {len(norm_windows):9,d} "
            f"{base_cad:11.2f} Hz {base_reg:9.3f} "
            f"{n_stumble:9d} {n_detected:9d}"
        )
        report_lines.append(line)

        # Save individual baseline using portable bundle format
        baseline.save(BASELINES_DIR / f"{subj}.joblib")

    # Fit a universal elderly default baseline for cold-start users
    elderly_norm = df[(df["cohort"] == "elderly") & (df["label"] == 0)]
    default_baseline = PersonalGaitBaseline(user_id="default_elderly")
    default_baseline.fit(elderly_norm)
    default_baseline.save(BASELINES_DIR / "default_elderly.joblib")

    # Fit a universal young adult baseline
    young_norm = df[(df["cohort"] == "young") & (df["label"] == 0)]
    young_baseline = PersonalGaitBaseline(user_id="default_young")
    young_baseline.fit(young_norm)
    young_baseline.save(BASELINES_DIR / "default_young.joblib")

    summary_msg = (
        f"\nPersonalized Anomaly Detection across all stumble trials: "
        f"{detected_stumbles}/{total_stumbles} detected ({detected_stumbles / max(1, total_stumbles):.1%})"
    )
    report_lines.append("-" * 70)
    report_lines.append(summary_msg)

    report_path = REPORTS / "personalization_report.txt"
    report_path.write_text("\n".join(report_lines))

    print("\n".join(report_lines[:20]))
    print("... [remaining subjects saved]")
    print(summary_msg)
    print(f"\nSaved {len(subjects)} baselines + defaults to {BASELINES_DIR}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
