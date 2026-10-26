import sqlite3, time
DB = "rollator.db"

def conn():
    return sqlite3.connect(DB)

def init():
    c = conn()
    try:
        with c:
            c.execute("""CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY, ts REAL, session TEXT, type TEXT,
                fall_prob REAL, instability REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS readings(
                id INTEGER PRIMARY KEY, ts REAL, session TEXT, state TEXT,
                fall_prob REAL, instability REAL, cadence_spm REAL,
                step_regularity REAL, mobility_score REAL)""")
    finally:
        c.close()

def log_event(session, etype, fall_prob, instability):
    c = conn()
    try:
        with c:
            c.execute("INSERT INTO events(ts,session,type,fall_prob,instability) VALUES(?,?,?,?,?)",
                      (time.time(), session, etype, fall_prob, instability))
    except sqlite3.OperationalError:
        init()
        with c:
            c.execute("INSERT INTO events(ts,session,type,fall_prob,instability) VALUES(?,?,?,?,?)",
                      (time.time(), session, etype, fall_prob, instability))
    finally:
        c.close()

def recent(n=50):
    c = conn()
    try:
        rows = c.execute("SELECT ts,session,type,fall_prob,instability FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(zip(["ts", "session", "type", "fall_prob", "instability"], r)) for r in rows]
    except sqlite3.OperationalError:
        init()
        rows = c.execute("SELECT ts,session,type,fall_prob,instability FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(zip(["ts", "session", "type", "fall_prob", "instability"], r)) for r in rows]
    finally:
        c.close()

def insert_reading(session, state, fall_prob, instability, cadence_spm=None, step_regularity=None, mobility_score=None, ts=None):
    if isinstance(session, (int, float)) and isinstance(state, str):
        real_ts = session
        session = state
        state = fall_prob
        fall_prob = instability
        instability = cadence_spm
        cadence_spm = step_regularity
        step_regularity = mobility_score
        mobility_score = ts
        ts = real_ts
    if ts is None:
        ts = time.time()
    c = conn()
    try:
        with c:
            c.execute("""INSERT INTO readings(ts, session, state, fall_prob, instability, cadence_spm, step_regularity, mobility_score)
                         VALUES(?,?,?,?,?,?,?,?)""",
                      (ts, session, state, fall_prob, instability, cadence_spm, step_regularity, mobility_score))
    except sqlite3.OperationalError:
        init()
        with c:
            c.execute("""INSERT INTO readings(ts, session, state, fall_prob, instability, cadence_spm, step_regularity, mobility_score)
                         VALUES(?,?,?,?,?,?,?,?)""",
                      (ts, session, state, fall_prob, instability, cadence_spm, step_regularity, mobility_score))
    finally:
        c.close()

def recent_readings(minutes=10, limit=600):
    cutoff = time.time() - (float(minutes) * 60) if minutes else 0.0
    c = conn()
    try:
        rows = c.execute("""
            SELECT id, ts, session, state, fall_prob, instability, cadence_spm, step_regularity, mobility_score
            FROM (
                SELECT id, ts, session, state, fall_prob, instability, cadence_spm, step_regularity, mobility_score
                FROM readings
                WHERE ts >= ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
        """, (cutoff, limit)).fetchall()
        cols = ["id", "ts", "session", "state", "fall_prob", "instability", "cadence_spm", "step_regularity", "mobility_score"]
        return [dict(zip(cols, r)) for r in rows]
    except sqlite3.OperationalError:
        init()
        return []
    finally:
        c.close()



