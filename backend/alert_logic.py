"""NORMAL -> WARNING -> FALL_PENDING -> ALERT_SENT, with debounce, 'I'm OK' cancel window and cooldown."""
NORMAL, WARNING, FALL_PENDING, ALERT_SENT = "NORMAL", "WARNING", "FALL_PENDING", "ALERT_SENT"

class AlertEngine:
    def __init__(self, fall_thr=0.8, fall_windows=2, unstable_thr=0.40, unstable_windows=3,
                 cancel_secs=15, cooldown_secs=30):
        self.fall_thr, self.fall_windows = fall_thr, fall_windows
        self.unstable_thr, self.unstable_windows = unstable_thr, unstable_windows
        self.cancel_secs, self.cooldown_secs = cancel_secs, cooldown_secs
        self.state, self._since = NORMAL, 0.0
        self._fall_count, self._unstable_count = 0, 0

    def update(self, fall_prob, instability, now):
        """Returns (state, event_or_None). event is logged / triggers notification."""
        event = None
        if self.state == FALL_PENDING:
            if now - self._since >= self.cancel_secs:
                self.state, self._since, event = ALERT_SENT, now, "fall_alert_sent"
        elif self.state == ALERT_SENT:
            if now - self._since >= self.cooldown_secs:
                self.state = NORMAL
        else:
            # a fall needs N consecutive high-probability windows (cuts single-window false alarms)
            self._fall_count = self._fall_count + 1 if fall_prob >= self.fall_thr else 0
            if self._fall_count >= self.fall_windows:
                self.state, self._since, event = FALL_PENDING, now, "fall_detected"
                self._fall_count = 0
            else:
                self._unstable_count = self._unstable_count + 1 if instability >= self.unstable_thr else 0
                if self._unstable_count >= self.unstable_windows:
                    if self.state != WARNING:
                        event = "instability_warning"
                    self.state = WARNING
                elif self._unstable_count == 0:
                    self.state = NORMAL
        return self.state, event

    def cancel(self, now):
        if self.state == FALL_PENDING:
            self.state, self._since = NORMAL, now
            return "fall_cancelled_by_user"