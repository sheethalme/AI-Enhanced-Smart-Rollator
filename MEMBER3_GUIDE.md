# Member 3 Guide: App, Integration & Hardware (phone-first)

You own the **live system**: phone sensor -> backend -> models -> alerts -> caregiver view. Until hardware exists, the phone *is* the sensor. Later, the ESP32 sends the **same messages**, so nothing else changes.

## 1. Architecture you are building

```
Phone browser (sender.html)           Backend (FastAPI)                       Outputs
 devicemotion -> g, deg/s  --WS-->  StreamBuffer (resample 50 Hz, 100-sample windows)
 batches every 0.5 s                     |-> predict_fall()  (Member 1; stub now)
                                         |-> predict_gait()  (Member 2; stub now)
                                         v
                                   AlertEngine (state machine) --> SQLite events
                                         |                     --> notify_caregiver() (Telegram)
 status + "I'm OK"  <----WS-----         +--> /api/events --> caregiver dashboard
                                         +--> CSV recorder --> data/own_captures/ (for Member 2)
```

Hardware swap later = replace the phone page with ESP32 firmware that sends `{"samples": [[t_ms, ax, ay, az, gx, gy, gz], ...]}` to `/ws/stream`. The backend never knows the difference.

## 2. What's already built (starter code)

| File | What it does |
|---|---|
| `backend/main.py` | WebSocket `/ws/stream`, resampling to 50 Hz, windowing, model calls, alert engine, optional CSV recording, `/api/events` |
| `backend/stubs.py` | Fake `predict_fall` / `predict_gait` with the **contract signatures**. Replace bodies when real models arrive |
| `backend/alert_logic.py` | NORMAL -> WARNING -> FALL_PENDING (15 s cancel window) -> ALERT_SENT, with debounce and cooldown |
| `backend/db.py` | SQLite event log |
| `backend/notify.py` | Placeholder; becomes a Telegram bot in Week 5 |
| `backend/static/sender.html` | Phone page: streams motion data, label picker, record toggle, live state, "I'm OK" button |

It was smoke-tested with simulated data (a spike correctly triggers `FALL_PENDING` and logs the event). The stubs are placeholders, not real detection.

## 3. Setup (Windows / PowerShell)

```powershell
cd member3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. Getting the phone to connect (the one tricky part)

Phone motion sensors only work on **HTTPS** pages, and plain `http://<laptop-ip>:8000` will not work. Use a tunnel:

```powershell
# option A (no account needed)
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
# option B
ngrok http 8000
```
Open the printed `https://...` URL on your phone -> pick a label -> **Start streaming**.

Notes:
- **iPhone (Safari):** a permission popup appears when you tap Start. Accept it.
- **Android (Chrome):** works directly. Check Settings -> Site settings -> Motion sensors is allowed.
- iOS and Android report accelerometer axes with **opposite signs**. Harmless because models use signal magnitude, but tell Member 1 and 2 which phone was used.
- Keep the screen on while streaming (browsers pause sensors in the background).

## 5. Your workflow, step by step

**Week 1: Foundation (do this first)**
1. Run the starter code and confirm the phone streams and `/api/events` works.
2. With the team, freeze `docs/data_contract.md` and `docs/model_contract.md` (window size, units, function signatures). Check the phone page matches: g, deg/s, ms timestamps, 50 Hz output.
3. Create the repo and folders; add `.gitignore` for `*.db`, `data/raw`, `.venv`.
4. **Immediately help Member 2**: tell them the CSV recorder exists. Tick "Record CSV", pick the label, and the file lands in `data/own_captures/` in the contract format. Your streaming tool is their data-collection tool.

**Week 2: Harden the stream**
- Handle dropped connections (auto-reconnect in `sender.html`), show connection status.
- Log actual sample rate received; if the phone delivers less than ~30 Hz, note it.
- Add a `/api/status` endpoint (current state, last fall_prob and instability, uptime).

**Week 3: Live end-to-end**
- Walk with the phone taped to a chair/walker frame and confirm state changes appear on the phone in real time.
- Measure end-to-end latency (phone timestamp vs server receive time). Target under 1 s.

**Week 4: Integrate real models (checkpoint #1)**
- In `stubs.py`, import Member 1's `predict_fall` and Member 2's `predict_gait`. Keep stubs as a fallback via an environment flag (`USE_STUBS=1`).
- Tune `AlertEngine` thresholds with real outputs. Agree the thresholds with Members 1 and 2.
- Add a test folder with a replay script that streams a recorded CSV through the backend, so you can test without walking each time.

**Week 5: Caregiver side**
- Build the Telegram bot notification in `notify.py` (create a bot with @BotFather, send messages with `requests.post`).
- Build the caregiver dashboard (Streamlit is fastest): live state, alert history from `/api/events`, daily walking activity.
- Wire in Member 2's `get_mobility_report(user_id, days)` for the trend views.

**Week 6: Trends + hardware prep (checkpoint #2)**
- Persist daily metrics in SQLite (walking minutes, cadence, variability, health score).
- Start ESP32 + MPU6050 firmware in a simulator first (Wokwi) sending the same JSON. Reuse your ESP32/MQTT experience from the IoT project.

**Week 7: Hardware swap + full testing**
- Flash the ESP32, mount on the rollator, confirm identical behaviour to the phone.
- Note MPU6050 range settings (+/-8 g, +/-500 deg/s recommended) and convert to g and deg/s in firmware.

**Week 8: Demo**
- Scripted demo: normal walk -> unstable walk (warning) -> simulated fall -> "I'm OK" cancel -> real alert to Telegram -> dashboard trend.
- Record a backup video. Draw the final architecture diagram.

## 6. How to test each piece

| What | How |
|---|---|
| Streaming | Watch the state text on the phone update every ~1 s |
| Recording | Check a CSV appears in `data/own_captures/` with columns `timestamp,ax,ay,az,gx,gy,gz,label` |
| Alerts | Shake the phone hard once: expect `FALL_PENDING`, then tap "I'm OK" to cancel or wait 15 s for `ALERT_SENT` |
| Events | Open `/api/events` in the browser |
| Replay | Post a recorded CSV through the websocket with a small Python script (Week 4) |

## 7. Working rules for you

- You only edit `backend/`, `app/`, `firmware/`. Anything in `models/` or `personalization/` is read-only for you; ask for changes via a PR or message.
- Never change the contract signatures in `stubs.py`; if one has to change, raise it with the team.
- Ask Members 1 and 2 early for: exported model files, `requirements`, and a 5-line usage example each.

## 8. Known limits to mention in the report
- Phone browsers throttle sensors in the background and vary in real sample rate (25-60 Hz); we resample to 50 Hz.
- Alert thresholds are tuned on simulated and volunteer data, not elderly users.
- Fall alerts are an assistive early warning, not a certified safety device.
