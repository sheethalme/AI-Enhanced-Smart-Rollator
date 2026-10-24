# Member 2: Gait Analysis, Personalization & Longitudinal Decline Tracker

AI-Enhanced Smart Rollator module responsible for real-time walking stability classification, individual elderly gait personalization, and longitudinal mobility decline tracking.

---

## 1. System Architecture

```
[6-Axis IMU Stream: ax, ay, az (g) | gx, gy, gz (deg/s) @ 50 Hz]
                              │
                              ▼
        [125-sample / 2.5-second Sliding Buffer (50% overlap)]
                              │
                              ▼
        [02_gait_features.py: 26 Biomechanical Gait Features]
            ├── Cadence (FFT peak in 0.8-3.0 Hz band & SPM)
            ├── Step Regularity & Symmetry (Autocorrelation lags)
            ├── Stride Timing & Variability (Peak intervals, CV %)
            ├── Mediolateral Sway & Sway Ratio (Lateral RMS / Vertical RMS)
            └── Dynamic Smoothness & Energy (Jerk RMS, SMA, Gyro sway)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    [Tier 1: Global Instability]     [Tier 2: Personalization]
     03_train_instability.py           04_personalization.py
     Random Forest Classifier          Individual Isolation Forest
     (Normal vs Stumble / Trip)        (Learns personal norm from walks)
              │                               │
              └───────────────┬───────────────┘
                              ▼
            [05_decline_tracker.py: Caregiver Layer]
            ├── Mobility Health Score (0 - 100)
            ├── Longitudinal Multi-Week Drift Tracking
            └── Caregiver Early-Warning Alert ('stable' | 'declining' | 'improving')
                              │
                              ▼
            [predict_gait.py: Member 3 Deployable API]
            ├── predict_gait(window, user_id=None)
            └── get_mobility_report(user_id, days=14)
```

---

## 2. Directory Structure

```
m2-gait-features/
├── data/
│   ├── processed/
│   │   ├── gait_windows.parquet       # 17,902 harmonized windows (125 samples @ 50 Hz)
│   │   └── gait_features.parquet      # 17,902 rows x 26 biomechanical features
│   └── reports/
│       ├── instability_training_report.txt
│       ├── personalization_report.txt
│       ├── mobility_decline_report.json
│       └── mobility_decline_trend.png # 4-panel longitudinal trend visualization
├── models/
│   └── gait/
│       ├── gait_rf.pkl                # Production Random Forest instability classifier
│       ├── config.json                # Model hyperparameters, metrics & feature list
│       └── baselines/                 # Individual user baseline bundles (*.joblib)
├── scripts/
│   ├── common_gait.py                 # Shared paths, constants & PersonalGaitBaseline
│   ├── 01_harmonize_gait.py           # SisFall & MobiFall harmonization to 50 Hz
│   ├── 02_gait_features.py            # 26 Biomechanical feature extractor
│   ├── 03_train_instability.py        # GroupKFold (subject-independent) training
│   ├── 04_personalization.py          # Per-user Isolation Forest baseline fitting
│   └── 05_decline_tracker.py          # Mobility score & 4-week drift simulation
├── predict_gait.py                    # Deployable inference API (Member 3 contract)
├── requirements.txt                   # Dependency definitions
└── README.md                          # Module documentation
```

---

## 3. The 26 Biomechanical Gait Features

Rather than using generic statistical aggregations, Member 2 extracts 26 clinically validated biomechanical gait features:

| Category | Features | Clinical & Biomechanical Rationale |
|---|---|---|
| **Cadence & Spectrum** | `cadence_fft_hz`, `cadence_spm`, `spectral_power_gait`, `spectral_entropy` | Dominant step frequency in normal 0.8–3.0 Hz band; spectral concentration measures rhythmicity. |
| **Regularity & Symmetry** | `step_regularity`, `stride_regularity`, `stride_symmetry`, `harmonic_ratio` | Autocorrelation peaks at step and stride lags. Harmonic ratio measures smooth cyclical harmonic motion vs erratic hesitations. |
| **Stride Variability (Fall Risk)** | `step_count`, `stride_time_mean`, `stride_time_std`, `stride_variability_cv`, `step_peak_intensity_mean`, `step_peak_intensity_std`, `step_intensity_cv` | Step interval and impact force coefficient of variation (CV %). High CV (>12%) is the gold standard clinical predictor of geriatric falls. |
| **Balance & Lateral Sway** | `lateral_sway_rms`, `lateral_sway_range`, `vertical_rms`, `anteroposterior_rms`, `sway_ratio` | Excessive side-to-side wobble indicates poor lateral balance in rollator users. |
| **Smoothness & Motion Energy** | `acc_jerk_mean`, `acc_jerk_rms`, `acc_sma`, `gyro_sma`, `gyro_sway_rms`, `gmag_mean` | High acceleration jerk and erratic yaw rotation signify sudden stumbling or abrupt balance corrections. |

---

## 4. Evaluation & Results

### Tier 1: Global Instability Classifier (5-Fold GroupKFold by Subject)
- **Validation Protocol:** 5-Fold GroupKFold strictly grouped by subject (0 subject leakage).
- **ROC-AUC:** `0.9891`
- **Specificity:** `99.7%` (17,108 / 17,165 stable windows correctly cleared; < 0.3% false alarm rate)
- **Sensitivity:** `88.5%` (652 / 737 stumble events correctly detected on unseen subjects)
- **Precision:** `92.0%`
- **F1 Score:** `0.9018`

### Tier 2: Per-User Personalization (Isolation Forest Baselines)
- Fitted individual baselines on normal walking sessions for all 47 subjects (`SA01`-`SA23`, `SE01`-`SE15`, `mobi_01`-`mobi_09`).
- Anomaly detection evaluated on individual stumble trials: **91.7%** of stumbles flagged as significant personal anomalies.

### Longitudinal Mobility Health Tracking (4-Week Decline Drift)
- Simulates 28-day progressive elderly physical decline on subject `SE01`.
- Mobility Health Score drops from **94.1** (stable baseline) to **69.9** (marked frailty).
- Weekly slope: **-9.93 pts/week** -> Classification: `'DECLINING'`.
- Caregiver Alert: **Triggered** with clinical check recommendation before a fall occurs.

---

## 5. Deployable API (Member 3 Contract)

```python
from predict_gait import predict_gait, get_mobility_report
import numpy as np

# 1. Real-time window inference (125 samples @ 50 Hz, 6 channels)
# window shape: (125, 6) = [ax, ay, az, gx, gy, gz]
window = np.zeros((125, 6)) # replace with live 50 Hz sensor buffer
result = predict_gait(window, user_id="SE01")

print(result)
# Output:
# {
#   'instability_score': 0.0000,
#   'is_unstable': False,
#   'cadence_hz': 1.76,
#   'cadence_spm': 106.0,
#   'step_regularity': 0.728,
#   'stride_variability_cv': 2.9,
#   'lateral_sway_rms': 0.188,
#   'mobility_score': 99.8,
#   'personal_anomaly': {...},
#   'latency_ms': 108.5
# }

# 2. Historical Caregiver Mobility Report
report = get_mobility_report(user_id="SE01", days=14)
print(report)
# Output:
# {
#   'user_id': 'SE01',
#   'days_analyzed': 14,
#   'current_mobility_score': 69.9,
#   'trend': 'declining',
#   'slope_points_per_week': -9.93,
#   'caregiver_alert': True,
#   'recommendation': 'Warning: Progressive mobility decline detected. Caregiver check recommended.'
# }
```

---

## 6. How to Re-Run Everything End-to-End

```bash
cd m2-gait-features

# Step 1: Harmonize SisFall & MobiFall into 50 Hz windows
python scripts/01_harmonize_gait.py

# Step 2: Extract 26 biomechanical gait features
python scripts/02_gait_features.py

# Step 3: Train Random Forest Instability Classifier
python scripts/03_train_instability.py

# Step 4: Fit individual personal baselines
python scripts/04_personalization.py

# Step 5: Run longitudinal decline tracker & drift simulation
python scripts/05_decline_tracker.py

# Step 6: Test deployable API & smoke test
python predict_gait.py
```
