"""Extract 26 Biomechanical Gait Features from 50 Hz IMU windows.

Features cover:
- Cadence and frequency spectrum (FFT dominant frequency, power ratio, spectral entropy)
- Rhythmic regularity and symmetry (autocorrelation step & stride regularity, harmonic ratio)
- Stride timing and variability (peak intervals, coefficient of variation CV)
- Lateral balance and sway (RMS sway, sway range, sway-to-vertical ratio)
- Dynamic smoothness and intensity (jerk, SMA, angular velocity)

Outputs: data/processed/gait_features.parquet
"""
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import entropy

from common_gait import (
    FS, WIN, ACC_CLIP, GAIT_FEATURE_NAMES, META_COLS,
    PROCESSED
)


def compute_window_features(ax, ay, az, gx, gy, gz, clip=True, fs=FS):
    """Compute the 26 biomechanical gait features for a single 125-sample window.

    Parameters
    ----------
    ax, ay, az : 1D array of 125 floats (g)
    gx, gy, gz : 1D array of 125 floats (deg/s)
    clip : bool, clip acceleration to +/- 2g
    fs : float, sampling rate (default 50.0 Hz)

    Returns
    -------
    dict of 26 feature name-value pairs
    """
    # 1. Clip acceleration if required (matching hardware sensor limits)
    if clip:
        ax = np.clip(ax, -ACC_CLIP, ACC_CLIP)
        ay = np.clip(ay, -ACC_CLIP, ACC_CLIP)
        az = np.clip(az, -ACC_CLIP, ACC_CLIP)

    # 2. Magnitudes
    amag = np.sqrt(ax**2 + ay**2 + az**2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    n = len(amag)

    amag_det = amag - np.mean(amag)
    var_amag = float(np.var(amag_det))

    # 3. Frequency domain (Cadence & Spectral Power)
    # Zero-pad to 256 points for ~0.195 Hz frequency resolution
    if var_amag > 1e-5:
        fft_vals = np.abs(np.fft.rfft(amag_det, n=256))
        freqs = np.fft.rfftfreq(256, d=1.0 / fs)
        # Gait band: human walking step frequency is typically 0.8 - 3.0 Hz
        mask = (freqs >= 0.8) & (freqs <= 3.0)
        if mask.any() and fft_vals[mask].sum() > 1e-6:
            peak_idx = np.argmax(fft_vals[mask])
            cadence_fft_hz = float(freqs[mask][peak_idx])
            band_power = float(np.sum(fft_vals[mask]**2))
            total_power = float(np.sum(fft_vals**2)) + 1e-9
            spectral_power_gait = float(band_power / total_power)

            # Spectral entropy: measure of regularity / chaos
            psd_norm = fft_vals[1:] / (np.sum(fft_vals[1:]) + 1e-9)
            spectral_entropy = float(entropy(psd_norm + 1e-9))
        else:
            cadence_fft_hz = 0.0
            spectral_power_gait = 0.0
            spectral_entropy = 0.0
    else:
        cadence_fft_hz = 0.0
        spectral_power_gait = 0.0
        spectral_entropy = 0.0

    cadence_spm = float(cadence_fft_hz * 60.0)

    # 4. Autocorrelation & Step Regularity
    # Regular walking shows clear periodic correlation peaks
    if var_amag > 1e-5:
        ac = np.correlate(amag_det, amag_det, mode="full")[n - 1:]
        ac = ac / (ac[0] + 1e-9)
        # Search for first step peak between 0.3s and 1.2s (15 to 60 samples @ 50 Hz)
        search_ac = ac[15:min(61, len(ac))]
        if len(search_ac) > 0 and np.max(search_ac) > 0:
            step_lag = 15 + int(np.argmax(search_ac))
            step_regularity = float(np.clip(ac[step_lag], 0.0, 1.0))
            stride_lag = min(step_lag * 2, len(ac) - 1)
            stride_regularity = float(np.clip(ac[stride_lag], 0.0, 1.0))
        else:
            step_regularity = 0.0
            stride_regularity = 0.0
    else:
        step_regularity = 0.0
        stride_regularity = 0.0

    stride_symmetry = float(step_regularity / (stride_regularity + 1e-6))

    # 5. Harmonic Ratio (Menz et al. clinical index: even vs odd harmonics)
    if cadence_fft_hz > 0:
        fft_raw = np.abs(np.fft.rfft(amag_det, n=256))
        freqs_raw = np.fft.rfftfreq(256, d=1.0 / fs)
        evens, odds = [], []
        for h in range(1, 11):
            target_f = h * cadence_fft_hz
            idx = int(np.argmin(np.abs(freqs_raw - target_f)))
            amp = fft_raw[idx]
            if h % 2 == 0:
                evens.append(amp)
            else:
                odds.append(amp)
        harmonic_ratio = float(np.sum(evens) / (np.sum(odds) + 1e-6))
    else:
        harmonic_ratio = 0.0

    # 6. Peak Detection & Stride Variability (CV)
    # Minimum 0.3s between consecutive steps (15 samples @ 50 Hz)
    peaks, _ = find_peaks(amag, distance=15, prominence=0.10)
    step_count = len(peaks)
    if step_count >= 2:
        step_intervals = np.diff(peaks) / fs  # in seconds
        stride_time_mean = float(np.mean(step_intervals))
        stride_time_std = float(np.std(step_intervals))
        stride_variability_cv = float((stride_time_std / (stride_time_mean + 1e-6)) * 100.0)

        peak_heights = amag[peaks]
        step_peak_intensity_mean = float(np.mean(peak_heights))
        step_peak_intensity_std = float(np.std(peak_heights))
        step_intensity_cv = float((step_peak_intensity_std / (step_peak_intensity_mean + 1e-6)) * 100.0)
    else:
        stride_time_mean = 0.0
        stride_time_std = 0.0
        stride_variability_cv = 0.0
        step_peak_intensity_mean = float(np.mean(amag[peaks])) if step_count == 1 else 0.0
        step_peak_intensity_std = 0.0
        step_intensity_cv = 0.0

    # 7. Balance & Lateral Sway (Side-to-side vs vertical)
    lateral_sway_rms = float(np.sqrt(np.mean(ax**2)))
    lateral_sway_range = float(np.ptp(ax))
    vertical_rms = float(np.sqrt(np.mean(az**2)))
    anteroposterior_rms = float(np.sqrt(np.mean(ay**2)))
    sway_ratio = float(lateral_sway_rms / (vertical_rms + 1e-6))

    # 8. Dynamic Smoothness & Energy (Jerk, SMA, Gyro)
    jerk = np.abs(np.diff(amag)) * fs
    acc_jerk_mean = float(np.mean(jerk))
    acc_jerk_rms = float(np.sqrt(np.mean(jerk**2)))
    acc_sma = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    gyro_sma = float(np.mean(np.abs(gx) + np.abs(gy) + np.abs(gz)))
    gyro_sway_rms = float(np.sqrt(np.mean(gx**2 + gy**2)))
    gmag_mean = float(np.mean(gmag))

    return {
        "cadence_fft_hz": cadence_fft_hz,
        "cadence_spm": cadence_spm,
        "spectral_power_gait": spectral_power_gait,
        "spectral_entropy": spectral_entropy,
        "step_regularity": step_regularity,
        "stride_regularity": stride_regularity,
        "stride_symmetry": stride_symmetry,
        "harmonic_ratio": harmonic_ratio,
        "step_count": float(step_count),
        "stride_time_mean": stride_time_mean,
        "stride_time_std": stride_time_std,
        "stride_variability_cv": stride_variability_cv,
        "step_peak_intensity_mean": step_peak_intensity_mean,
        "step_peak_intensity_std": step_peak_intensity_std,
        "step_intensity_cv": step_intensity_cv,
        "lateral_sway_rms": lateral_sway_rms,
        "lateral_sway_range": lateral_sway_range,
        "vertical_rms": vertical_rms,
        "anteroposterior_rms": anteroposterior_rms,
        "sway_ratio": sway_ratio,
        "acc_jerk_mean": acc_jerk_mean,
        "acc_jerk_rms": acc_jerk_rms,
        "acc_sma": acc_sma,
        "gyro_sma": gyro_sma,
        "gyro_sway_rms": gyro_sway_rms,
        "gmag_mean": gmag_mean,
    }


def extract_features_from_dataframe(df_windows):
    """Iterate through windows and compute the 26 biomechanical features."""
    print(f"Extracting features across {len(df_windows):,} windows...")
    t0 = time.time()

    # Pre-extract signal arrays
    all_ax = np.array(df_windows["ax"].tolist(), dtype=np.float64)
    all_ay = np.array(df_windows["ay"].tolist(), dtype=np.float64)
    all_az = np.array(df_windows["az"].tolist(), dtype=np.float64)
    all_gx = np.array(df_windows["gx"].tolist(), dtype=np.float64)
    all_gy = np.array(df_windows["gy"].tolist(), dtype=np.float64)
    all_gz = np.array(df_windows["gz"].tolist(), dtype=np.float64)

    rows = []
    n = len(df_windows)
    for i in range(n):
        feats = compute_window_features(
            all_ax[i], all_ay[i], all_az[i],
            all_gx[i], all_gy[i], all_gz[i],
            clip=True, fs=FS
        )
        rows.append(feats)

    df_feats = pd.DataFrame(rows)
    # Combine metadata columns and feature columns
    meta_df = df_windows[META_COLS].reset_index(drop=True)
    out_df = pd.concat([meta_df, df_feats], axis=1)

    elapsed = time.time() - t0
    print(f"Extracted {len(out_df):,} feature rows in {elapsed:.2f}s ({elapsed / len(out_df) * 1000:.2f} ms/window)")
    return out_df


def main():
    print("=" * 70)
    print("STEP 2: BIOMECHANICAL GAIT FEATURE EXTRACTION (26 Features)")
    print("=" * 70)

    in_path = PROCESSED / "gait_windows.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run 01_harmonize_gait.py first.")

    df_windows = pd.read_parquet(in_path)
    df_features = extract_features_from_dataframe(df_windows)

    # Save to parquet
    out_path = PROCESSED / "gait_features.parquet"
    df_features.to_parquet(out_path, index=False)

    print("\n" + "=" * 70)
    print("BIOMECHANICAL FEATURE CONTRAST: STABLE vs UNSTABLE GAIT")
    print("=" * 70)
    comp_feats = [
        "cadence_fft_hz", "step_regularity", "stride_variability_cv",
        "lateral_sway_rms", "sway_ratio", "acc_jerk_mean", "harmonic_ratio"
    ]
    summary = df_features.groupby("label_name")[comp_feats].mean().T
    print(summary.to_string())

    print(f"\nSuccessfully wrote {len(df_features):,} rows x {len(GAIT_FEATURE_NAMES)} features to:")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
