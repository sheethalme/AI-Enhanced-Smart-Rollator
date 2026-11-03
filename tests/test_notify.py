"""Unit tests for Telegram caregiver alerts in backend.notify."""
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import httpx
from backend import notify
from backend.notify import notify_caregiver


class TestNotify(unittest.TestCase):
    """Test suite verifying caregiver alert notifications via Telegram Bot API."""

    def setUp(self):
        # Save original environment variables
        self.orig_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.orig_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        # Ensure clean state
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        notify.reset_state()

    def tearDown(self):
        # Restore environment variables
        if self.orig_token is not None:
            os.environ["TELEGRAM_BOT_TOKEN"] = self.orig_token
        else:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)

        if self.orig_chat_id is not None:
            os.environ["TELEGRAM_CHAT_ID"] = self.orig_chat_id
        else:
            os.environ.pop("TELEGRAM_CHAT_ID", None)

        notify.reset_state()

    @patch("httpx.post")
    def test_a_no_env_vars(self, mock_post):
        """(a) no env vars -> no post call, no exception, last_notification.ok is False."""
        captured_stdout = io.StringIO()
        try:
            with patch("sys.stdout", captured_stdout):
                t = notify_caregiver("fall_alert_sent", {"fall_prob": 0.85})
                if t:
                    t.join(timeout=2.0)
        except Exception as exc:
            self.fail(f"notify_caregiver raised unexpected exception when unconfigured: {exc}")

        # Verify no network request was made
        mock_post.assert_not_called()

        # Verify console output occurred as before
        output = captured_stdout.getvalue()
        self.assertIn("[NOTIFY CAREGIVER]", output)
        self.assertIn("FALL ALERT", output)

        # Verify last_notification state
        last = notify.last_notification
        self.assertIsNotNone(last)
        self.assertFalse(last["ok"])
        self.assertEqual(last["reason"], "not configured")
        self.assertEqual(last["event"], "fall_alert_sent")
        self.assertIsInstance(last["ts"], float)

    @patch("httpx.post")
    def test_b_env_vars_set(self, mock_post):
        """(b) env vars set -> post called once with the right URL and chat_id."""
        token = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz_1234567"
        chat_id = "987654321"
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        os.environ["TELEGRAM_CHAT_ID"] = chat_id

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        t = notify_caregiver("fall_alert_sent", {"fall_prob": 0.95})
        if t:
            t.join(timeout=2.0)

        # Verify post called once
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args

        # Right URL
        expected_url = f"https://api.telegram.org/bot{token}/sendMessage"
        url = call_args[0] if call_args else call_kwargs.get("url")
        self.assertEqual(url, expected_url)

        # Right chat_id and message format
        data = call_kwargs.get("data", {})
        self.assertEqual(data.get("chat_id"), chat_id)
        text = data.get("text", "")
        self.assertIn("FALL ALERT: possible fall detected at", text)
        self.assertIn("Fall probability 0.95", text)
        self.assertIn("Please check on the user.", text)

        # Timeout 5 s
        self.assertEqual(call_kwargs.get("timeout"), 5.0)

        # Verify last_notification
        last = notify.last_notification
        self.assertIsNotNone(last)
        self.assertTrue(last["ok"])
        self.assertEqual(last["reason"], "ok")
        self.assertEqual(last["event"], "fall_alert_sent")
        self.assertIsInstance(last["ts"], float)

    @patch("httpx.post")
    def test_c_post_raises(self, mock_post):
        """(c) post raises -> notify_caregiver does not raise and last_notification.ok is False."""
        token = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz_1234567"
        chat_id = "987654321"
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        os.environ["TELEGRAM_CHAT_ID"] = chat_id

        # Network error simulation
        mock_post.side_effect = httpx.ConnectError("Network unreachable")

        try:
            t = notify_caregiver("fall_alert_sent", {"fall_prob": 0.88})
            if t:
                t.join(timeout=2.0)
        except Exception as exc:
            self.fail(f"notify_caregiver must never raise on network failure, but raised: {exc}")

        last = notify.last_notification
        self.assertIsNotNone(last)
        self.assertFalse(last["ok"])
        self.assertIn("Network unreachable", last["reason"])
        self.assertEqual(last["event"], "fall_alert_sent")

        # Also test HTTP non-200 (e.g. 401 Unauthorized for bad token)
        notify.reset_state()
        bad_token_resp = MagicMock()
        bad_token_resp.status_code = 401
        mock_post.side_effect = None
        mock_post.return_value = bad_token_resp

        t2 = notify_caregiver("fall_alert_sent", {"fall_prob": 0.88})
        if t2:
            t2.join(timeout=2.0)

        last2 = notify.last_notification
        self.assertIsNotNone(last2)
        self.assertFalse(last2["ok"])
        self.assertEqual(last2["reason"], "HTTP 401")

    @patch("httpx.post")
    def test_d_token_never_appears_in_printed_output(self, mock_post):
        """(d) the token never appears in printed output or error logs."""
        secret_token = "TOP_SECRET_BOT_TOKEN_NEVER_LOG_ME_888"
        chat_id = "987654321"
        os.environ["TELEGRAM_BOT_TOKEN"] = secret_token
        os.environ["TELEGRAM_CHAT_ID"] = chat_id

        # Simulate exception where the URL in error text contains the token
        mock_post.side_effect = httpx.ConnectError(
            f"Failed to connect to https://api.telegram.org/bot{secret_token}/sendMessage"
        )

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        with patch("sys.stdout", captured_stdout), patch("sys.stderr", captured_stderr):
            t = notify_caregiver("fall_alert_sent", {"fall_prob": 0.90})
            if t:
                t.join(timeout=2.0)

        out = captured_stdout.getvalue()
        err = captured_stderr.getvalue()

        # Token must NOT appear in stdout or stderr
        self.assertNotIn(secret_token, out)
        self.assertNotIn(secret_token, err)

        # Token must NOT appear in last_notification["reason"]
        last = notify.last_notification
        self.assertIsNotNone(last)
        self.assertNotIn(secret_token, str(last["reason"]))
        self.assertIn("[REDACTED]", str(last["reason"]))

    @patch("httpx.post")
    def test_instability_warning_message_format(self, mock_post):
        """Verify instability_warning message text and delivery."""
        token = "12345:TOKEN"
        chat_id = "555"
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        os.environ["TELEGRAM_CHAT_ID"] = chat_id

        mock_resp = MagicMock(status_code=200)
        mock_post.return_value = mock_resp

        t = notify_caregiver("instability_warning", {"instability": 0.55})
        if t:
            t.join(timeout=2.0)

        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        data = call_kwargs.get("data", {})
        self.assertEqual(data.get("chat_id"), chat_id)
        text = data.get("text", "")
        self.assertIn("Unsteady walking detected at", text)
        self.assertTrue(notify.last_notification["ok"])
        self.assertEqual(notify.last_notification["event"], "instability_warning")

    @patch("httpx.post")
    def test_minimum_gaps_rate_limiting(self, mock_post):
        """Verify minimum gaps: 30 s between fall alerts, 300 s between instability warnings."""
        token = "12345:TOKEN"
        chat_id = "555"
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        os.environ["TELEGRAM_CHAT_ID"] = chat_id

        mock_resp = MagicMock(status_code=200)
        mock_post.return_value = mock_resp

        # 1. Fall alerts minimum gap (30 s)
        t1 = notify_caregiver("fall_alert_sent", {"fall_prob": 0.90})
        if t1:
            t1.join(timeout=2.0)
        self.assertEqual(mock_post.call_count, 1)

        # Second call immediately after should be suppressed
        t2 = notify_caregiver("fall_alert_sent", {"fall_prob": 0.90})
        if t2:
            t2.join(timeout=2.0)
        self.assertEqual(mock_post.call_count, 1)

        # 2. Instability warnings minimum gap (300 s)
        t3 = notify_caregiver("instability_warning", {"instability": 0.45})
        if t3:
            t3.join(timeout=2.0)
        self.assertEqual(mock_post.call_count, 2)

        # Second call immediately after should be suppressed
        t4 = notify_caregiver("instability_warning", {"instability": 0.45})
        if t4:
            t4.join(timeout=2.0)
        self.assertEqual(mock_post.call_count, 2)

    def test_status_endpoint_notifier_fields(self):
        """Verify /api/status returns last_notification and notifier_configured without exposing secrets."""
        from backend.main import status

        # Unconfigured
        data = status()
        self.assertIn("notifier_configured", data)
        self.assertFalse(data["notifier_configured"])
        self.assertIn("last_notification", data)
        self.assertIsNone(data["last_notification"])

        # With env vars configured
        token = "SECRET_TOKEN_DO_NOT_EXPOSE"
        chat_id = "SECRET_CHAT_ID"
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        os.environ["TELEGRAM_CHAT_ID"] = chat_id

        data2 = status()
        self.assertTrue(data2["notifier_configured"])

        # Secrets must never appear in response
        data_str = str(data2)
        self.assertNotIn(token, data_str)
        self.assertNotIn(chat_id, data_str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
