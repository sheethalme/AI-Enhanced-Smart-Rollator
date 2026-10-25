"""Integration test for Member 3 WebSocket stream and prediction pipeline."""
import os
import gc
import glob
from pathlib import Path
import sys
import unittest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from fastapi.testclient import TestClient
from backend.main import app


class TestBackendStream(unittest.TestCase):
    """Test suite for backend /ws/stream WebSocket endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls._cleanup()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup()

    def tearDown(self):
        self._cleanup()

    @staticmethod
    def _cleanup():
        """Ensure no rollator.db or generated CSV files remain after tests."""
        gc.collect()

        # Clean rollator.db in repo root or current directory
        db_paths = [Path("rollator.db"), Path(__file__).resolve().parents[1] / "rollator.db"]
        for db_path in db_paths:
            if db_path.exists():
                try:
                    db_path.unlink()
                except OSError:
                    pass

        # Clean any stray CSV files created in data/own_captures
        captures_dir = Path(__file__).resolve().parents[1] / "data" / "own_captures"
        if captures_dir.exists():
            for csv_file in captures_dir.glob("*.csv"):
                try:
                    csv_file.unlink()
                except OSError:
                    pass

    def test_stream_synthetic_samples(self):
        """Send ~4 seconds of 100 Hz synthetic sensor data with a 3.5g spike and verify response keys."""
        # 4 seconds at 100 Hz = 400 samples (10 ms interval)
        samples = []
        for i in range(400):
            t_ms = float(i * 10)  # 10 ms interval -> 100 Hz
            ax = 0.0
            ay = 0.0
            az = 3.5 if i == 200 else 1.0  # az around 1.0 with one spike of 3.5 g
            gx = 0.0
            gy = 0.0
            gz = 0.0
            samples.append([t_ms, ax, ay, az, gx, gy, gz])

        with self.client.websocket_connect("/ws/stream?record=0") as ws:
            ws.send_json({"samples": samples})
            reply = ws.receive_json()

            self.assertIsInstance(reply, dict)
            self.assertIn("state", reply)
            self.assertIn("fall_prob", reply)
            self.assertIn("instability", reply)

            # Validate field types and values
            self.assertIsInstance(reply["state"], str)
            self.assertIsInstance(reply["fall_prob"], (float, int))
            self.assertIsInstance(reply["instability"], (float, int))
            self.assertGreaterEqual(reply["fall_prob"], 0.0)
            self.assertGreaterEqual(reply["instability"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
