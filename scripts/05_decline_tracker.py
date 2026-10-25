"""Longitudinal Mobility Decline Tracker & Clinical Drift Simulation.

Provides:
1. Mobility Health Score (0-100) combining cadence, regularity, variability, sway, and smoothness.
2. Trajectory trend detection: 'stable', 'declining', 'improving'.
3. 4-week clinical drift simulation demonstrating caregiver early-warning alerts.
4. Generates trend plots and reports in data/reports/.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Set non-interactive Matplotlib backend and local cache directory
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / ".mplcache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_gait import (
    PROCESSED, REPORTS, BASELINES_DIR, GAIT_FEATURE_NAMES,
    PersonalGaitBaseline
)


def compute_mobility_score(features, baseline=None):
    """Compute composite Mobility Health Score (0 - 100).

    Components:
    - Cadence (25 pts): walking speed and rhythmicity
    - Regularity (25 pts): consistency of step-to-step timing
    - Stride Variability (25 pts): coefficient of variation (strong fall risk indicator)
    - Lateral Balance (25 pts): side-to-side stability and sway control
    """
    if isinstance(features, (pd.DataFrame, pd.Series)):
        f = features.to_dict()
    elif isinstance(features, dict):
        f = features
    else:
        f = {k: v for k, v in zip(GAIT_FEATURE_NAMES, features)}

    cadence = float(f.get("cadence_fft_hz", 1.5))
    step_reg = float(f.get("step_regularity", 0.5))
    stride_cv = float(f.get("stride_variability_cv", 8.0))
    sway_rms = float(f.get("lateral_sway_rms", 0.20))
    jerk = float(f.get("acc_jerk_mean", 4.0))

    if baseline is not None and hasattr(baseline, "baseline_stats") and baseline.baseline_stats:
        # Personalized baseline scoring
        b_cad = baseline.baseline_stats["cadence_fft_hz"]["mean"]
        b_reg = baseline.baseline_stats["step_regularity"]["mean"]
        b_cv = baseline.baseline_stats["stride_variability_cv"]["mean"]
        b_sway = baseline.baseline_stats["lateral_sway_rms"]["mean"]

        # Sub-scores (0 to 25 each)
        # Cadence: ratio relative to personal norm
        cad_ratio = cadence / (b_cad + 1e-6)
        s_cad = 25.0 * np.clip(1.0 - abs(1.0 - cad_ratio) * 1.8, 0.0, 1.0)

        # Regularity: ratio relative to personal norm
        reg_ratio = step_reg / (b_reg + 1e-6)
        s_reg = 25.0 * np.clip(reg_ratio, 0.0, 1.2) / 1.2

        # Stride Variability CV: penalized if increases above personal norm
        cv_ratio = stride_cv / (b_cv + 1e-6)
        s_cv = 25.0 * np.clip(1.0 - max(0.0, cv_ratio - 1.0) * 0.75, 0.0, 1.0)

        # Sway: penalized if lateral sway increases above personal norm
        sway_ratio = sway_rms / (b_sway + 1e-6)
        s_sway = 25.0 * np.clip(1.0 - max(0.0, sway_ratio - 1.0) * 0.75, 0.0, 1.0)
    else:
        # Population geriatric normative scoring (geriatric clinical references)
        # Healthy cadence: ~1.4 - 1.9 Hz (84 - 114 steps/min)
        if 1.4 <= cadence <= 2.0:
            s_cad = 25.0
        elif cadence > 0.5:
            s_cad = 25.0 * max(0.0, 1.0 - abs(cadence - 1.65) * 1.2)
        else:
            s_cad = 5.0

        # Step regularity: ideal > 0.6, frail < 0.3
        s_reg = 25.0 * np.clip(step_reg / 0.70, 0.0, 1.0)

        # Stride variability: ideal < 6%, moderate 8-12%, high fall risk > 15%
        if stride_cv <= 6.0:
            s_cv = 25.0
        else:
            s_cv = 25.0 * np.clip(1.0 - (stride_cv - 6.0) / 14.0, 0.0, 1.0)

        # Lateral sway: ideal < 0.20 g, high wobble > 0.35 g
        if sway_rms <= 0.20:
            s_sway = 25.0
        else:
            s_sway = 25.0 * np.clip(1.0 - (sway_rms - 0.20) / 0.20, 0.0, 1.0)

    # Minor jerk smoothness deduction if movement is abrupt
    jerk_penalty = min(5.0, max(0.0, (jerk - 6.0) * 0.5))

    total_score = float(np.clip(s_cad + s_reg + s_cv + s_sway - jerk_penalty, 0.0, 100.0))
    return round(total_score, 1)


class MobilityHealthTracker:
    """Tracks daily mobility metrics and detects longitudinal physical decline."""

    def __init__(self, user_id="SE01", baseline=None):
        self.user_id = user_id
        self.baseline = baseline
        self.daily_records = []

    def record_day(self, day_num, cadence, regularity, stride_cv, sway_rms, active_minutes, n_windows=50):
        """Record aggregated metrics for a single day."""
        feats = {
            "cadence_fft_hz": cadence,
            "step_regularity": regularity,
            "stride_variability_cv": stride_cv,
            "lateral_sway_rms": sway_rms,
            "acc_jerk_mean": 3.5 + (stride_cv * 0.15),
        }
        score = compute_mobility_score(feats, baseline=self.baseline)

        record = {
            "day": int(day_num),
            "cadence_hz": round(float(cadence), 3),
            "step_regularity": round(float(regularity), 3),
            "stride_variability_cv": round(float(stride_cv), 2),
            "lateral_sway_rms": round(float(sway_rms), 3),
            "active_minutes": round(float(active_minutes), 1),
            "mobility_score": score,
        }
        self.daily_records.append(record)
        return record

    def analyze_trend(self, window_days=14):
        """Analyze trajectory over the past window_days.

        Returns
        -------
        dict:
            - trend: 'stable' | 'declining' | 'improving'
            - slope_points_per_week: float
            - score_change: float
            - alert_triggered: bool
            - recommendation: str
        """
        if len(self.daily_records) < 3:
            return {
                "trend": "insufficient_data",
                "slope_points_per_week": 0.0,
                "score_change": 0.0,
                "alert_triggered": False,
                "recommendation": "Collect more daily walking sessions.",
            }

        recent = self.daily_records[-window_days:]
        days = np.array([r["day"] for r in recent])
        scores = np.array([r["mobility_score"] for r in recent])

        # Linear regression slope (points per day)
        slope, intercept = np.polyfit(days, scores, 1)
        slope_per_week = float(slope * 7.0)
        score_change = float(scores[-1] - scores[0])
        current_score = float(scores[-1])

        # Clinical thresholds:
        # A drop of > 3 points/week or current score < 60 triggers caregiver alert
        alert = bool(slope_per_week < -3.0 or (current_score < 60.0 and score_change < 0))

        if slope_per_week <= -2.5:
            trend = "declining"
            rec = "Warning: Progressive mobility decline detected. Caregiver check recommended."
        elif slope_per_week >= 2.5:
            trend = "improving"
            rec = "Positive mobility recovery observed. Maintain regular walking exercises."
        else:
            trend = "stable"
            rec = "Mobility pattern is stable within normal personal range."

        return {
            "trend": trend,
            "slope_points_per_week": round(slope_per_week, 2),
            "score_change": round(score_change, 1),
            "current_score": current_score,
            "alert_triggered": alert,
            "recommendation": rec,
        }


def simulate_4week_drift(user_id="SE01"):
    """Simulate 4 weeks (28 days) of gradual elderly mobility decline."""
    print(f"\nSimulating 4-week clinical decline trajectory for subject '{user_id}'...")

    # Load personal baseline if available
    baseline_file = BASELINES_DIR / f"{user_id}.joblib"
    if baseline_file.exists():
        baseline = PersonalGaitBaseline.load(baseline_file)
        b_cad = baseline.baseline_stats["cadence_fft_hz"]["mean"]
        b_reg = baseline.baseline_stats["step_regularity"]["mean"]
        b_cv = baseline.baseline_stats["stride_variability_cv"]["mean"]
        b_sway = baseline.baseline_stats["lateral_sway_rms"]["mean"]
    else:
        baseline = None
        b_cad, b_reg, b_cv, b_sway = 1.70, 0.40, 9.5, 0.22

    tracker = MobilityHealthTracker(user_id=user_id, baseline=baseline)

    np.random.seed(42)
    days_data = []

    for d in range(1, 29):
        # Clinical drift parameters:
        # Week 1 (Days 1-7): Stable baseline
        if d <= 7:
            drift_factor = 0.0
        # Week 2 (Days 8-14): Subtle early fatigue (drift 0 to 0.25)
        elif d <= 14:
            drift_factor = (d - 7) / 7.0 * 0.25
        # Week 3 (Days 15-21): Progressive frailty (drift 0.25 to 0.65)
        elif d <= 21:
            drift_factor = 0.25 + (d - 14) / 7.0 * 0.40
        # Week 4 (Days 22-28): Marked clinical decline (drift 0.65 to 1.0)
        else:
            drift_factor = 0.65 + (d - 21) / 7.0 * 0.35

        # Inject realistic physiological drift + daily random noise:
        # Cadence drops by 14% at full drift
        cad = b_cad * (1.0 - 0.14 * drift_factor) + np.random.normal(0, 0.02)
        # Step regularity decreases by 35%
        reg = b_reg * (1.0 - 0.35 * drift_factor) + np.random.normal(0, 0.015)
        # Stride variability increases by 35%
        cv = b_cv * (1.0 + 0.35 * drift_factor) + np.random.normal(0, 0.4)
        # Lateral sway increases by 28%
        sway = b_sway * (1.0 + 0.28 * drift_factor) + np.random.normal(0, 0.008)
        # Active daily walking minutes decreases as frailty sets in
        active_mins = 25.0 * (1.0 - 0.30 * drift_factor) + np.random.normal(0, 2.0)

        rec = tracker.record_day(d, cad, reg, cv, sway, active_mins)
        days_data.append(rec)

    trend_eval = tracker.analyze_trend(window_days=14)
    print("\n" + "=" * 70)
    print("4-WEEK LONGITUDINAL DECLINE ANALYSIS")
    print("=" * 70)
    print(f"User ID:               {user_id}")
    print(f"Day 1 Mobility Score:  {days_data[0]['mobility_score']}")
    print(f"Day 28 Mobility Score: {days_data[-1]['mobility_score']} (Total drop: {days_data[-1]['mobility_score'] - days_data[0]['mobility_score']:.1f} pts)")
    print(f"Trend Classification:  '{trend_eval['trend'].upper()}'")
    print(f"Weekly Score Slope:    {trend_eval['slope_points_per_week']:.2f} pts/week")
    print(f"Caregiver Alert:       {'TRIGGERED [WARNING]' if trend_eval['alert_triggered'] else 'NONE'}")
    print(f"Caregiver Message:     {trend_eval['recommendation']}")

    # Generate 4-panel visual report
    plot_path = REPORTS / "mobility_decline_trend.png"
    generate_trend_plot(days_data, trend_eval, plot_path)

    # Save JSON report
    report_dict = {
        "user_id": user_id,
        "duration_days": 28,
        "baseline_summary": {
            "cadence_hz": round(b_cad, 3),
            "step_regularity": round(b_reg, 3),
            "stride_variability_cv": round(b_cv, 2),
            "lateral_sway_rms": round(b_sway, 3),
        },
        "trend_evaluation": trend_eval,
        "daily_history": days_data,
    }
    json_path = REPORTS / "mobility_decline_report.json"
    json_path.write_text(json.dumps(report_dict, indent=2))

    return report_dict


def generate_trend_plot(days_data, trend_eval, output_path):
    """Render and save publication-grade 4-panel longitudinal trend plot."""
    days = [d["day"] for d in days_data]
    scores = [d["mobility_score"] for d in days_data]
    cadence = [d["cadence_hz"] for d in days_data]
    variability = [d["stride_variability_cv"] for d in days_data]
    sway = [d["lateral_sway_rms"] for d in days_data]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=130)
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: Mobility Health Score
    ax1 = axes[0, 0]
    ax1.plot(days, scores, color="#1976D2", lw=2.5, marker="o", markersize=4, label="Daily Mobility Score")
    ax1.axhline(60, color="#D32F2F", ls="--", lw=1.5, label="Caregiver Alert Threshold (60)")
    ax1.axvspan(15, 28, color="#FFEBEE", alpha=0.5, label="Clinical Frailty Phase")
    ax1.set_title("A. Composite Mobility Health Score (0-100)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Health Score")
    ax1.set_ylim(40, 105)
    ax1.legend(loc="lower left", fontsize=8)

    # Panel 2: Cadence Degradation
    ax2 = axes[0, 1]
    ax2.plot(days, cadence, color="#388E3C", lw=2, marker="s", markersize=3)
    ax2.set_title("B. Cadence (Step Frequency, Hz)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Frequency (Hz)")

    # Panel 3: Stride Variability Escalation
    ax3 = axes[1, 0]
    ax3.plot(days, variability, color="#F57C00", lw=2, marker="^", markersize=4)
    ax3.axhline(12.0, color="#D32F2F", ls=":", label="Elevated Fall Risk (>12%)")
    ax3.set_title("C. Stride-to-Stride Timing Variability (CV %)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Coefficient of Variation (%)")
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Lateral Sway Wobble Escalation
    ax4 = axes[1, 1]
    ax4.plot(days, sway, color="#7B1FA2", lw=2, marker="d", markersize=4)
    ax4.set_title("D. Mediolateral Sway RMS (Side-to-Side Instability, g)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Lateral Sway (g)")

    plt.suptitle(
        f"Smart Rollator Longitudinal Mobility Tracking — 4-Week Decline Simulation\n"
        f"Detected Trend: '{trend_eval['trend'].upper()}' | Weekly Drift: {trend_eval['slope_points_per_week']:.1f} pts/wk",
        fontsize=12, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved trend visualization to: {output_path}")


def main():
    print("=" * 70)
    print("STEP 5: LONGITUDINAL DECLINE TRACKER & DRIFT SIMULATION")
    print("=" * 70)

    # Run simulation for elderly subject SE01
    simulate_4week_drift(user_id="SE01")


if __name__ == "__main__":
    main()
