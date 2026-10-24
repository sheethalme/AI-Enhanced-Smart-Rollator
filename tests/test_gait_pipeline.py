"""Unit and integration test suite for Member 2 Gait Module."""
import json
import unittest
from pathlib import Path
import numpy as np

# Add parent directory to path
import sys
MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(MODULE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(MODULE_DIR / "scripts"))

from common_gait import (
    FS, WIN, RAW_CHANNELS, GAIT_FEATURE_NAMES,
    MODELS_DIR, BASELINES_DIR, REPORTS,
    PersonalGaitBaseline
)
from predict_gait import predict_gait, get_mobility_report


class TestGaitPipeline(unittest.TestCase):
    """Test suite verifying gait feature extraction, models, and deployment APIs."""

    def test_01_window_contract_validation(self):
        """Test valid and invalid window shapes and numerical bounds."""
        # Valid shape
        w_valid = np.zeros((WIN, len(RAW_CHANNELS)))
        w_valid[:, 2] = 1.0
        res = predict_gait(w_valid)
        self.assertIn("instability_score", res)
        self.assertIn("is_unstable", res)
        self.assertIn("cadence_hz", res)
        self.assertIn("mobility_score", res)
        self.assertFalse(res["is_unstable"])

        # Invalid shape: too few samples
        with self.assertRaises(ValueError):
            predict_gait(np.zeros((100, 6)))

        # Invalid shape: too few channels
        with self.assertRaises(ValueError):
            predict_gait(np.zeros((125, 5)))

        # Invalid values: NaN
        w_nan = np.zeros((125, 6))
        w_nan[10, 0] = np.nan
        with self.assertRaises(ValueError):
            predict_gait(w_nan)

        # Invalid values: Inf
        w_inf = np.zeros((125, 6))
        w_inf[10, 0] = np.inf
        with self.assertRaises(ValueError):
            predict_gait(w_inf)

    def test_02_feature_integrity(self):
        """Verify all 26 biomechanical features are present and finite."""
        w = np.random.randn(125, 6) * 0.1
        w[:, 2] += 1.0
        import importlib.util
        spec = importlib.util.spec_from_file_location("feat_mod", MODULE_DIR / "scripts" / "02_gait_features.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        feats = mod.compute_window_features(
            w[:, 0], w[:, 1], w[:, 2],
            w[:, 3], w[:, 4], w[:, 5]
        )
        self.assertEqual(len(feats), 26)
        for k in GAIT_FEATURE_NAMES:
            self.assertIn(k, feats)
            self.assertTrue(np.isfinite(feats[k]), f"Feature {k} was not finite")

    def test_03_artifacts_exist(self):
        """Verify all required model binaries and configuration artifacts exist."""
        rf_path = MODELS_DIR / "gait_rf.pkl"
        config_path = MODELS_DIR / "config.json"
        baseline_path = BASELINES_DIR / "SE01.joblib"
        default_base_path = BASELINES_DIR / "default_elderly.joblib"
        plot_path = REPORTS / "mobility_decline_trend.png"
        report_path = REPORTS / "mobility_decline_report.json"

        self.assertTrue(rf_path.exists(), f"Missing {rf_path}")
        self.assertTrue(config_path.exists(), f"Missing {config_path}")
        self.assertTrue(baseline_path.exists(), f"Missing {baseline_path}")
        self.assertTrue(default_base_path.exists(), f"Missing {default_base_path}")
        self.assertTrue(plot_path.exists(), f"Missing {plot_path}")
        self.assertTrue(report_path.exists(), f"Missing {report_path}")

        # Check config json content
        conf = json.loads(config_path.read_text())
        self.assertEqual(conf["model_type"], "RandomForestClassifier")
        self.assertEqual(len(conf["feature_names"]), 26)
        self.assertGreater(conf["evaluation_metrics"]["roc_auc"], 0.95)

    def test_04_personal_baseline_scoring(self):
        """Verify PersonalGaitBaseline loads and evaluates anomaly scores."""
        baseline = PersonalGaitBaseline.load(BASELINES_DIR / "SE01.joblib")
        self.assertTrue(baseline.is_fitted)

        # Test normal window
        w_norm = np.zeros(26)
        for i, k in enumerate(GAIT_FEATURE_NAMES):
            w_norm[i] = baseline.baseline_stats[k]["mean"]

        res = baseline.score_window(w_norm)
        self.assertIn("anomaly_score", res)
        self.assertIn("is_anomalous", res)
        self.assertIn("deviations", res)
        self.assertLess(res["anomaly_score"], 0.60)

    def test_05_mobility_reporting_api(self):
        """Verify get_mobility_report API returns required keys and valid data."""
        report = get_mobility_report(user_id="SE01", days=14)
        self.assertIn("user_id", report)
        self.assertIn("current_mobility_score", report)
        self.assertIn("trend", report)
        self.assertIn("caregiver_alert", report)
        self.assertIn("recommendation", report)
        self.assertIn(report["trend"], ["stable", "declining", "improving"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
