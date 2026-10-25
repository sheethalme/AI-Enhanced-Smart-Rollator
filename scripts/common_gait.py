"""Shared constants, paths, and signal utilities for Member 2 gait analysis."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Directory paths
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SISFALL = PROJECT_ROOT / "data" / "sisfall"
RAW_MOBIFALL = PROJECT_ROOT / "data" / "mobifall"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports"
MODELS_DIR = ROOT / "models" / "gait"
BASELINES_DIR = MODELS_DIR / "baselines"

# Sampling and windowing parameters (Shared team contract)
FS = 50                         # Sampling rate in Hz
DT = 1.0 / FS                   # Sample duration (0.02 s)
WIN = 125                       # 2.5 seconds window (125 samples @ 50 Hz)
STRIDE = 62                     # ~50% overlap stride (1.24 s)
ACC_CLIP = 2.0                  # Standard clipping limit in g (+/- 2.0g)
G = 9.81                        # Gravity acceleration (m/s^2)
RAD2DEG = 57.2958               # Radians to degrees conversion factor

# Channel specifications
RAW_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "amag", "gmag"]

# Gait stability labels
STABLE_WALK = 0                 # Normal steady gait (D01, D02, WAL, JOG)
UNSTABLE_WALK = 1               # Stumbles, slips, loss of balance, near-falls (D18, D11)
LABEL_NAMES = {STABLE_WALK: "stable", UNSTABLE_WALK: "unstable"}

# Sensor conversion scales for SisFall
SISFALL_ACC_SCALE = 32.0 / 8192.0
SISFALL_GYRO_SCALE = 4000.0 / 65536.0
SISFALL_FS = 200.0

# Activities to harmonize
SISFALL_ACTIVITIES = ["D01", "D02", "D11", "D12", "D15", "D17", "D18"]
MOBIFALL_ACTIVITIES = ["WAL", "JOG"]

# The 26 Biomechanical Gait & Stability Features
GAIT_FEATURE_NAMES = [
    "cadence_fft_hz",
    "cadence_spm",
    "spectral_power_gait",
    "spectral_entropy",
    "step_regularity",
    "stride_regularity",
    "stride_symmetry",
    "harmonic_ratio",
    "step_count",
    "stride_time_mean",
    "stride_time_std",
    "stride_variability_cv",
    "step_peak_intensity_mean",
    "step_peak_intensity_std",
    "step_intensity_cv",
    "lateral_sway_rms",
    "lateral_sway_range",
    "vertical_rms",
    "anteroposterior_rms",
    "sway_ratio",
    "acc_jerk_mean",
    "acc_jerk_rms",
    "acc_sma",
    "gyro_sma",
    "gyro_sway_rms",
    "gmag_mean",
]

META_COLS = ["dataset", "subject", "cohort", "activity", "trial", "start", "label", "label_name"]
KEY_MONITORED_METRICS = [
    "cadence_fft_hz",
    "step_regularity",
    "stride_variability_cv",
    "lateral_sway_rms",
    "acc_jerk_mean",
    "sway_ratio"
]


def resample_uniform(t_raw, values, t_start=None, t_end=None, fs_target=FS):
    """Interpolate irregular or multi-rate time series onto a uniform 50 Hz grid."""
    dt_target = 1.0 / fs_target
    if t_start is None:
        t_start = t_raw[0]
    if t_end is None:
        t_end = t_raw[-1]

    t_grid = np.arange(t_start, t_end, dt_target)
    if len(t_grid) == 0:
        return np.array([]), np.empty((0, values.shape[1] if values.ndim > 1 else 1))

    if values.ndim == 1:
        v_grid = np.interp(t_grid, t_raw, values)
    else:
        v_grid = np.column_stack([np.interp(t_grid, t_raw, values[:, j]) for j in range(values.shape[1])])
    return t_grid, v_grid


def derive_signal_magnitudes(ax, ay, az, gx, gy, gz, clip=True):
    """Compute amag and gmag, and optionally clip acceleration to +/-2g."""
    if clip:
        ax = np.clip(ax, -ACC_CLIP, ACC_CLIP)
        ay = np.clip(ay, -ACC_CLIP, ACC_CLIP)
        az = np.clip(az, -ACC_CLIP, ACC_CLIP)
    amag = np.sqrt(ax**2 + ay**2 + az**2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    return ax, ay, az, gx, gy, gz, amag, gmag


class PersonalGaitBaseline:
    """Encapsulates a user's personal walking profile and anomaly detector."""

    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=0.05,
            n_estimators=100,
            random_state=42
        )
        self.feature_names = GAIT_FEATURE_NAMES
        self.baseline_stats = {}
        self.is_fitted = False

    def fit(self, X_normal):
        """Fit baseline on normal walking windows."""
        if isinstance(X_normal, pd.DataFrame):
            X_df = X_normal[self.feature_names]
            X_mat = X_df.to_numpy()
        else:
            X_mat = np.asarray(X_normal, dtype=np.float64)
            X_df = pd.DataFrame(X_mat, columns=self.feature_names)

        # 1. Fit scaler & Isolation Forest
        X_scaled = self.scaler.fit_transform(X_mat)
        self.model.fit(X_scaled)

        # 2. Store baseline statistical profile
        self.baseline_stats = {}
        for col in self.feature_names:
            series = X_df[col].to_numpy()
            self.baseline_stats[col] = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "median": float(np.median(series)),
                "p5": float(np.percentile(series, 5)),
                "p95": float(np.percentile(series, 95)),
            }
        self.is_fitted = True
        return self

    def score_window(self, features):
        """Evaluate a window's features against the user's personal baseline."""
        if not self.is_fitted:
            raise RuntimeError("Baseline has not been fitted.")

        if isinstance(features, (dict, pd.Series)):
            vec = np.array([[features[k] for k in self.feature_names]], dtype=np.float64)
        elif isinstance(features, (list, np.ndarray)):
            vec = np.asarray(features, dtype=np.float64).reshape(1, -1)
        else:
            raise ValueError("features must be dict, pd.Series, list, or array")

        # Isolation Forest score: raw score_samples is negative; normalize to [0, 1]
        raw_score = self.model.score_samples(self.scaler.transform(vec))[0]
        norm_score = float(np.clip(1.0 / (1.0 + np.exp(8.0 * (raw_score + 0.50))), 0.0, 1.0))
        is_anomaly = bool(self.model.predict(self.scaler.transform(vec))[0] == -1)

        # Detailed metric deviations (Z-scores)
        deviations = {}
        for metric in KEY_MONITORED_METRICS:
            idx = self.feature_names.index(metric)
            val = float(vec[0, idx])
            m_mean = self.baseline_stats[metric]["mean"]
            m_std = self.baseline_stats[metric]["std"]
            z = float((val - m_mean) / (m_std + 1e-6))
            deviations[metric] = {
                "current_val": round(val, 3),
                "baseline_mean": round(m_mean, 3),
                "z_score": round(z, 2),
            }

        return {
            "user_id": self.user_id,
            "anomaly_score": round(norm_score, 4),
            "is_anomalous": is_anomaly,
            "deviations": deviations,
        }

    def save(self, path):
        """Serialize baseline bundle to disk."""
        bundle = {
            "user_id": self.user_id,
            "scaler": self.scaler,
            "model": self.model,
            "baseline_stats": self.baseline_stats,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path):
        """Deserialize baseline bundle from disk."""
        data = joblib.load(path)
        if isinstance(data, cls):
            return data
        obj = cls(user_id=data.get("user_id", "default"))
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.baseline_stats = data["baseline_stats"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted = data["is_fitted"]
        return obj
