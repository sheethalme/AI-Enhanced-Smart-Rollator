"""Integration and unit tests for caregiver dashboard and backend APIs."""
import os
import gc
from pathlib import Path
import sys
import unittest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from fastapi.testclient import TestClient
from backend.main import app, LATEST


class TestDashboardApi(unittest.TestCase):
    """Test suite for caregiver dashboard endpoints and WebSocket data flow."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls._cleanup()
        # Reset LATEST to initial NO_DATA state
        LATEST.update({
            "state": "NO_DATA",
            "fall_prob": None,
            "instability": None,
            "cadence_spm": None,
            "step_regularity": None,
            "mobility_score": None,
            "session": None,
            "last_update": None,
            "pending_since": None,
        })

    @classmethod
    def tearDownClass(cls):
        cls._cleanup()

    @staticmethod
    def _cleanup():
        """Ensure rollator.db and generated CSV files are cleaned up."""
        gc.collect()
        db_paths = [Path("rollator.db"), REPO_DIR / "rollator.db"]
        for db_path in db_paths:
            if db_path.exists():
                try:
                    db_path.unlink()
                except OSError:
                    pass

        captures_dir = REPO_DIR / "data" / "own_captures"
        if captures_dir.exists():
            for csv_file in captures_dir.glob("*.csv"):
                try:
                    csv_file.unlink()
                except OSError:
                    pass

    def test_01_initial_status_no_data(self):
        """GET /api/status before phone connects returns 200 and state NO_DATA."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("state", data)
        self.assertEqual(data["state"], "NO_DATA")
        self.assertIn("connected", data)
        self.assertFalse(data["connected"])
        self.assertIn("thresholds", data)
        self.assertEqual(data["thresholds"].get("fall"), 0.8)
        self.assertEqual(data["thresholds"].get("unstable"), 0.40)
        self.assertIsNone(data.get("seconds_since_update"))
        self.assertIsNone(data.get("pending_seconds_left"))

    def test_02_stream_and_api_endpoints(self):
        """Stream synthetic samples through /ws/stream, then verify all API endpoints."""
        # 400 samples at 100 Hz = 4 seconds
        samples = []
        for i in range(400):
            t_ms = float(i * 10)
            ax = 0.0
            ay = 0.0
            az = 3.5 if i == 200 else 1.0
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

        # 1. GET /api/status
        status_res = self.client.get("/api/status")
        self.assertEqual(status_res.status_code, 200)
        st = status_res.json()
        self.assertIn("state", st)
        self.assertIn("fall_prob", st)
        self.assertIn("instability", st)
        self.assertIn("cadence_spm", st)
        self.assertIn("step_regularity", st)
        self.assertIn("mobility_score", st)
        self.assertIn("session", st)
        self.assertIn("last_update", st)
        self.assertIn("connected", st)
        self.assertTrue(st["connected"])
        self.assertIn("thresholds", st)
        self.assertEqual(st["thresholds"]["fall"], 0.8)
        self.assertEqual(st["thresholds"]["unstable"], 0.40)
        self.assertIsNotNone(st.get("seconds_since_update"))

        # 2. GET /api/readings
        readings_res = self.client.get("/api/readings?minutes=10")
        self.assertEqual(readings_res.status_code, 200)
        readings = readings_res.json()
        self.assertIsInstance(readings, list)
        self.assertGreaterEqual(len(readings), 1)
        first_r = readings[0]
        self.assertIn("ts", first_r)
        self.assertIn("session", first_r)
        self.assertIn("state", first_r)
        self.assertIn("fall_prob", first_r)
        self.assertIn("instability", first_r)

        # 3. GET /api/events
        events_res = self.client.get("/api/events")
        self.assertEqual(events_res.status_code, 200)
        events = events_res.json()
        self.assertIsInstance(events, list)

        # 4. GET /api/mobility_report
        mob_res = self.client.get("/api/mobility_report")
        self.assertEqual(mob_res.status_code, 200)
        mob = mob_res.json()
        self.assertIsInstance(mob, dict)
        self.assertTrue(mob.get("simulated"))
        self.assertIn("user_id", mob)
        self.assertIn("trend", mob)
        self.assertIn("recommendation", mob)

        # 5. GET /dashboard
        dash_res = self.client.get("/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn("text/html", dash_res.headers.get("content-type", ""))

    def test_03_notify_caregiver_called_once_per_event(self):
        """Verify notify_caregiver and [NOTIFY CAREGIVER] output occurs once per alert event."""
        import io
        from unittest.mock import patch

        # Synthetic samples with high fluctuation to trigger instability_warning
        # 600 samples at 100 Hz = 6 seconds (enough for 3+ windows)
        samples = []
        for i in range(600):
            t_ms = float(i * 10)
            ax = 0.8 if (i % 4 == 0) else -0.8
            ay = 0.5 if (i % 2 == 0) else -0.5
            az = 1.0
            gx = 40.0 if (i % 2 == 0) else -40.0
            gy = 20.0
            gz = 0.0
            samples.append([t_ms, ax, ay, az, gx, gy, gz])

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            with self.client.websocket_connect("/ws/stream?record=0") as ws:
                ws.send_json({"samples": samples})
                reply = ws.receive_json()

        output = captured_output.getvalue()
        # Count "[NOTIFY CAREGIVER]" in the output
        notify_count = output.count("[NOTIFY CAREGIVER]")
        # It should be 1 if warning was triggered, or 0 if not; but never > 1 per event!
        # Check that there are no duplicate pairs
        lines = [line for line in output.splitlines() if "[NOTIFY CAREGIVER]" in line]
        # In case an event was triggered, verify every event type in lines is unique or matches events logged
        events = self.client.get("/api/events").json()
        warning_events = [e for e in events if e.get("type") in ("fall_alert_sent", "instability_warning")]
        self.assertEqual(len(lines), len(warning_events))


if __name__ == "__main__":
    unittest.main(verbosity=2)

