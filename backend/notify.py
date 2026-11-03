"""Caregiver notification module with Telegram bot alerts and rate-limiting."""
import os
import threading
import time
from typing import Any, Dict, Optional
import httpx

# Rate limiting minimum gaps (seconds)
MIN_GAPS: Dict[str, float] = {
    "fall_alert_sent": 30.0,
    "instability_warning": 300.0,
}

# Module-level state
last_notification: Optional[Dict[str, Any]] = None
_last_sent: Dict[str, float] = {}
_lock = threading.Lock()


def is_configured() -> bool:
    """Return True if both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in environment."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    return bool(token and chat_id)


def reset_state() -> None:
    """Helper to reset notification state (primarily for tests)."""
    global last_notification, _last_sent
    with _lock:
        last_notification = None
        _last_sent.clear()


def _sanitize(text: str, token: Optional[str]) -> str:
    """Ensure token never appears in any string, log, or error text."""
    if not token or not text:
        return text
    return text.replace(token, "[REDACTED]")


def format_message(event: str, details: Optional[Dict[str, Any]] = None) -> str:
    """Format caregiver alert message based on event type."""
    details = details or {}
    ts = details.get("ts")
    if ts is not None:
        local_time = time.strftime("%H:%M:%S", time.localtime(ts))
    else:
        local_time = time.strftime("%H:%M:%S")

    if event == "fall_alert_sent":
        fp = details.get("fall_prob", 0.0)
        x = f"{fp:.2f}" if isinstance(fp, (float, int)) else str(fp)
        return f"FALL ALERT: possible fall detected at {local_time}. Fall probability {x}. Please check on the user."
    elif event == "instability_warning":
        return f"Unsteady walking detected at {local_time}."
    else:
        return f"Alert: {event} detected at {local_time}."


def _send_telegram(token: str, chat_id: str, text: str, event: str) -> None:
    """Worker function executed inside a daemon thread."""
    global last_notification
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    ts = time.time()
    try:
        resp = httpx.post(url, data=payload, timeout=5.0)
        if resp.status_code == 200:
            with _lock:
                last_notification = {
                    "ok": True,
                    "reason": "ok",
                    "ts": ts,
                    "event": event,
                }
        else:
            with _lock:
                last_notification = {
                    "ok": False,
                    "reason": f"HTTP {resp.status_code}",
                    "ts": ts,
                    "event": event,
                }
    except Exception as e:
        safe_err = _sanitize(str(e), token)
        with _lock:
            last_notification = {
                "ok": False,
                "reason": safe_err,
                "ts": ts,
                "event": event,
            }


def notify_caregiver(
    event: str,
    details: Optional[Dict[str, Any]] = None,
    _sync: bool = False
) -> Optional[threading.Thread]:
    """
    Send caregiver alert via Telegram or print to console if unconfigured.
    Enforces minimum gaps: 30 s between fall alerts, 300 s between instability warnings.
    Executes in a daemon thread so the WebSocket loop is never blocked.
    """
    global last_notification, _last_sent
    details = details or {}
    now = time.time()

    # Enforce minimum gaps between alerts
    min_gap = MIN_GAPS.get(event)
    if min_gap is not None:
        with _lock:
            last_t = _last_sent.get(event)
            if last_t is not None and (now - last_t) < min_gap:
                return None
            _last_sent[event] = now

    msg_text = format_message(event, details)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        # Print the message to the console as now
        print(f"[NOTIFY CAREGIVER] {msg_text}")
        with _lock:
            last_notification = {
                "ok": False,
                "reason": "not configured",
                "ts": now,
                "event": event,
            }
        return None

    if _sync:
        _send_telegram(token, chat_id, msg_text, event)
        return None

    t = threading.Thread(
        target=_send_telegram,
        args=(token, chat_id, msg_text, event),
        daemon=True,
    )
    t.start()
    return t
