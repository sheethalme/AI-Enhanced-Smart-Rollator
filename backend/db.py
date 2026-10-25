import sqlite3, time
DB = "rollator.db"

def conn():
    return sqlite3.connect(DB)

def init():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY, ts REAL, session TEXT, type TEXT,
            fall_prob REAL, instability REAL)""")

def log_event(session, etype, fall_prob, instability):
    with conn() as c:
        c.execute("INSERT INTO events(ts,session,type,fall_prob,instability) VALUES(?,?,?,?,?)",
                  (time.time(), session, etype, fall_prob, instability))

def recent(n=50):
    with conn() as c:
        rows = c.execute("SELECT ts,session,type,fall_prob,instability FROM events ORDER BY id DESC LIMIT ?", (n,))
        return [dict(zip(["ts", "session", "type", "fall_prob", "instability"], r)) for r in rows]
