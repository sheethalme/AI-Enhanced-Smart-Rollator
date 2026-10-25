import json
import time
import csv
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from . import db
from .alert_logic import AlertEngine
from .notify import notify_caregiver
from .stubs import predict_fall, predict_gait


# ============================================================
# CONFIGURATION
# ============================================================

# 2.5 second window at 50 Hz
# 50% overlap
WIN = 125
STEP = 62
HZ = 50

STATIC = Path(__file__).parent / "static"

CAPTURES = (
    Path(__file__).parent.parent
    / "data"
    / "own_captures"
)

# Make sure the capture directory exists
CAPTURES.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI()

db.init()


# ============================================================
# SENSOR BUFFER
# ============================================================

class StreamBuffer:
    """
    Converts irregular phone sensor samples into a uniform
    50 Hz signal and creates sliding windows.

    Phone sample format:

    [
        timestamp_ms,
        ax,
        ay,
        az,
        gx,
        gy,
        gz
    ]
    """

    def __init__(self):
        self.raw = []
        self.next_t = None
        self.buf = np.empty((0, 6))

    def add(self, rows):
        """
        Add incoming phone samples.

        Returns:
            out  -> uniformly sampled sensor data
            wins -> completed sliding windows
        """

        self.raw.extend(rows)

        arr = np.array(self.raw, dtype=float)

        # Need at least two samples for interpolation
        if len(arr) < 2:
            return np.empty((0, 6)), []

        # First column is timestamp in milliseconds
        t = arr[:, 0]

        # Initialize interpolation time
        if self.next_t is None:
            self.next_t = t[0]

        # Create uniform 50 Hz time grid
        grid = np.arange(
            self.next_t,
            t[-1],
            1000 / HZ
        )

        if len(grid) == 0:
            return np.empty((0, 6)), []

        # Interpolate six sensor channels
        out = np.column_stack([
            np.interp(
                grid,
                t,
                arr[:, i]
            )
            for i in range(1, 7)
        ])

        # Update next interpolation timestamp
        self.next_t = (
            grid[-1] +
            1000 / HZ
        )

        # Keep the last raw sample for continuity
        self.raw = arr[
            t >= grid[-1]
        ].tolist()

        # Add uniformly sampled data to buffer
        self.buf = np.vstack([
            self.buf,
            out
        ])

        # Create sliding windows
        wins = []

        while len(self.buf) >= WIN:

            wins.append(
                self.buf[:WIN].copy()
            )

            # 50% overlap
            self.buf = self.buf[STEP:]

        return out, wins


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def sender():
    """
    Serve the phone sensor webpage.
    """

    return FileResponse(
        STATIC / "sender.html"
    )


# ============================================================
# RECENT EVENTS API
# ============================================================

@app.get("/api/events")
def events():
    """
    Return recent fall/instability events.
    """

    return db.recent()


# ============================================================
# WEBSOCKET SENSOR STREAM
# ============================================================

@app.websocket("/ws/stream")
async def stream(
    ws: WebSocket,
    label: str = "",
    record: int = 0
):
    """
    Receive phone sensor data through WebSocket.

    Query parameters:

        label=normal
        record=1

    Example:

        /ws/stream?label=normal&record=1
    """

    # --------------------------------------------------------
    # ACCEPT CONNECTION
    # --------------------------------------------------------

    await ws.accept()

    session = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    print(
        f"[STREAM] New session: {session}"
    )

    print(
        f"[STREAM] Label: {label}"
    )

    print(
        f"[STREAM] Recording: {bool(record)}"
    )

    # --------------------------------------------------------
    # CREATE SESSION OBJECTS
    # --------------------------------------------------------

    sb = StreamBuffer()

    engine = AlertEngine()

    f = None
    writer = None

    # Used to remember the values associated with
    # a detected fall before the alert is finally sent.
    trigger = (0, 0)

    # Number of uniformly sampled rows
    n = 0

    # --------------------------------------------------------
    # CREATE CSV FILE IF RECORDING IS ENABLED
    # --------------------------------------------------------

    if record:

        filename = (
            f"{label or 'unlabelled'}_"
            f"{session}.csv"
        )

        filepath = CAPTURES / filename

        f = open(
            filepath,
            "w",
            newline=""
        )

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
            "label"
        ])

        print(
            f"[CSV] Recording to: {filepath}"
        )

    # ========================================================
    # RECEIVE SENSOR DATA
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # RECEIVE MESSAGE FROM PHONE
            # ------------------------------------------------

            raw_message = await ws.receive_text()

            msg = json.loads(
                raw_message
            )

            now = time.time()

            # ------------------------------------------------
            # HANDLE "I'M OK" BUTTON
            # ------------------------------------------------

            if msg.get("cmd") == "ok":

                print(
                    "[STREAM] Cancel alert requested"
                )

                ev = engine.cancel(now)

                if ev:

                    db.log_event(
                        session,
                        ev,
                        0,
                        0
                    )

                # Tell phone that the alert was cancelled
                await ws.send_json({
                    "state": "normal",
                    "event": "alert_cancelled",
                    "fall_prob": 0,
                    "instability": 0
                })

                continue

            # ------------------------------------------------
            # CHECK FOR SENSOR SAMPLES
            # ------------------------------------------------

            if "samples" not in msg:

                print(
                    "[WARNING] Message does not "
                    "contain 'samples'"
                )

                continue

            samples = msg["samples"]

            # ------------------------------------------------
            # PROCESS SENSOR DATA
            # ------------------------------------------------

            out, wins = sb.add(
                samples
            )

            # ------------------------------------------------
            # SAVE RAW/RESAMPLED DATA TO CSV
            # ------------------------------------------------

            if writer:

                for row in out:

                    writer.writerow([
                        n / HZ,
                        *np.round(
                            row,
                            5
                        ),
                        label
                    ])

                    n += 1

                # Flush so data is actually written
                # while the session is running.
                f.flush()

            # ------------------------------------------------
            # PROCESS EACH COMPLETE WINDOW
            # ------------------------------------------------

            for w in wins:

                # ============================================
                # FALL PREDICTION
                # ============================================

                fall_result = predict_fall(w)

                fp = fall_result[
                    "fall_prob"
                ]

                # ============================================
                # GAIT / INSTABILITY PREDICTION
                # ============================================

                gait_result = predict_gait(w)

                ins = gait_result[
                    "instability_score"
                ]

                # ============================================
                # ALERT ENGINE
                # ============================================

                state, ev = engine.update(
                    fp,
                    ins,
                    now
                )

                print(
                    f"[PREDICTION] "
                    f"state={state}, "
                    f"event={ev}, "
                    f"fall_prob={fp:.3f}, "
                    f"instability={ins:.3f}"
                )

                # ============================================
                # IMPORTANT:
                # SEND STATUS BACK TO PHONE
                # ============================================

                await ws.send_json({
                    "state": state,
                    "event": ev,
                    "fall_prob": float(fp),
                    "instability": float(ins)
                })

                # ============================================
                # REMEMBER FALL VALUES
                # ============================================

                if ev == "fall_detected":

                    trigger = (
                        fp,
                        ins
                    )

                # ============================================
                # HANDLE EVENTS
                # ============================================

                if ev:

                    # If final alert is being sent,
                    # use the original fall values.
                    if ev == "fall_alert_sent":

                        lfp, lins = trigger

                    else:

                        lfp, lins = (
                            fp,
                            ins
                        )

                    # ----------------------------------------
                    # SAVE EVENT TO DATABASE
                    # ----------------------------------------

                    db.log_event(
                        session,
                        ev,
                        lfp,
                        lins
                    )

                    # ----------------------------------------
                    # NOTIFY CAREGIVER
                    # ----------------------------------------

                    if ev in (
                        "fall_alert_sent",
                        "instability_warning"
                    ):

                        notify_caregiver(
                            ev,
                            {
                                "session": session,
                                "fall_prob": float(lfp),
                                "instability": float(lins)
                            }
                        )

                        print(
                            "[NOTIFY CAREGIVER]",
                            ev,
                            {
                                "session": session,
                                "fall_prob": float(lfp),
                                "instability": float(lins)
                            }
                        )

    # ========================================================
    # PHONE DISCONNECTED
    # ========================================================

    except WebSocketDisconnect:

        print(
            f"[STREAM] "
            f"Phone disconnected: {session}"
        )

    # ========================================================
    # OTHER ERRORS
    # ========================================================

    except Exception as e:

        print(
            f"[STREAM ERROR] {e}"
        )

        # Try to inform the phone if the connection
        # is still available.
        try:

            await ws.send_json({
                "state": "error",
                "event": "server_error",
                "message": str(e)
            })

        except Exception:
            pass

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if f:

            f.close()

            print(
                f"[CSV] Closed recording: "
                f"{session}"
            )

        print(
            f"[STREAM] Session ended: "
            f"{session}"
        )