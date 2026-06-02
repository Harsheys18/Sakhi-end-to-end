"""
sakhi_db.py
===========
PostgreSQL storage layer for SAKHI robot.
Works with any hosted PostgreSQL — Neon, Supabase, Railway, or local.

Four public functions only:
  connect()   — create connection from DATABASE_URL env variable
  init_db()   — create tables (safe to call on every boot)
  put_in()    — write data into the database
  get_from()  — read data from the database

Setup:
  Set DATABASE_URL before running:

  PowerShell (cloud):
    $env:DATABASE_URL = "postgresql://user:pass@host/db?sslmode=require"

  PowerShell (local):
    $env:DATABASE_URL = "postgresql://sakhi:sakhi@localhost/sakhi"

  Linux / macOS:
    export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
"""

import os
import json
import time
from typing import Any, Optional
import psycopg2
import psycopg2.extras

psycopg2.extras.register_default_jsonb(loads=json.loads)


# ─────────────────────────────────────────────────────────────────────────────
#  VALID KEYS
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_SECTIONS = {
    "session_meta", "persons", "group_profile",
    "joke_tracker", "topics_discussed", "flags", "interaction_log",
}
_CURRENT_STATE_SECTIONS = {
    "snapshot", "vision", "audio", "fused", "robot", "table_context",
}


# ─────────────────────────────────────────────────────────────────────────────
#  connect()
# ─────────────────────────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    """
    Create a psycopg2 connection from the DATABASE_URL environment variable.

    PowerShell : $env:DATABASE_URL = "postgresql://user:pass@host/db?sslmode=require"
    Linux/macOS: export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL is not set.\n"
            "Example: postgresql://user:password@host/dbname?sslmode=require"
        )
    return psycopg2.connect(url)


# ─────────────────────────────────────────────────────────────────────────────
#  init_db()
# ─────────────────────────────────────────────────────────────────────────────

def init_db(conn) -> None:
    """
    Create all tables and indexes. Safe to call on every boot (IF NOT EXISTS).
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_memory (
                session_id       TEXT        PRIMARY KEY,
                session_meta     JSONB       NOT NULL DEFAULT '{}',
                persons          JSONB       NOT NULL DEFAULT '[]',
                group_profile    JSONB       NOT NULL DEFAULT '{}',
                joke_tracker     JSONB       NOT NULL DEFAULT '{}',
                topics_discussed JSONB       NOT NULL DEFAULT '[]',
                flags            JSONB       NOT NULL DEFAULT '{}',
                created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interaction_log (
                id               BIGSERIAL   PRIMARY KEY,
                session_id       TEXT        NOT NULL,
                seq              INT,
                ts_elapsed_min   FLOAT,
                entry_type       TEXT,
                intent           TEXT,
                response         TEXT,
                speaker_id       TEXT,
                target           TEXT,
                who_reacted_most TEXT,
                data             JSONB       NOT NULL,
                written_at       TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ilog_session ON interaction_log (session_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ilog_session_elapsed ON interaction_log (session_id, ts_elapsed_min);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ilog_type ON interaction_log (session_id, entry_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ilog_data_gin ON interaction_log USING GIN (data);")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS current_state (
                id                     BIGSERIAL   PRIMARY KEY,
                session_id             TEXT        NOT NULL,
                ts_ms                  BIGINT      NOT NULL,
                elapsed_min            FLOAT,
                group_energy           FLOAT,
                group_valence          FLOAT,
                dominant_quadrant      TEXT,
                attention_on_robot     FLOAT,
                laughter_detected      BOOLEAN,
                anyone_distressed      BOOLEAN,
                natural_pause_detected BOOLEAN,
                face_count             INT,
                snapshot               JSONB       NOT NULL,
                written_at             TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS ix_cs_session_ts ON current_state (session_id, ts_ms DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_cs_session_elapsed ON current_state (session_id, elapsed_min DESC);")
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  put_in()
# ─────────────────────────────────────────────────────────────────────────────

def put_in(conn, schema: str, section: str, data: Any, session_id: str) -> None:
    """
    Write data into the database.

    Parameters
    ----------
    conn       : psycopg2 connection  (from connect())
    schema     : "session_memory"  or  "current_state"
    section    : section name (see below)
    data       : value to store
    session_id : e.g. "ses_abc123"

    session_memory sections
    -----------------------
    "session_meta"      dict              merged into existing meta
    "persons"           dict | list[dict] upserted by id — new fields win, old kept
    "group_profile"     dict              full overwrite
    "joke_tracker"      dict              full overwrite
    "topics_discussed"  dict              appends one topic
                        list              replaces all topics
    "flags"             dict              merged — only given keys updated
    "interaction_log"   dict              appended as a new row (never overwrites)

    current_state sections
    ----------------------
    "snapshot"          dict              full current_state_v1 document → new row
    (other sections are read-only; always write via "snapshot")
    """

    # ── validation ────────────────────────────────────────────────────────────
    if schema not in ("session_memory", "current_state"):
        raise KeyError(f"Unknown schema '{schema}'. Use 'session_memory' or 'current_state'.")
    if schema == "session_memory" and section not in _SESSION_SECTIONS:
        raise KeyError(f"Unknown section '{section}'. Valid: {sorted(_SESSION_SECTIONS)}")
    if schema == "current_state" and section not in _CURRENT_STATE_SECTIONS:
        raise KeyError(f"Unknown section '{section}'. Valid: {sorted(_CURRENT_STATE_SECTIONS)}")
    if schema == "current_state" and section != "snapshot":
        raise ValueError("current_state: write via section='snapshot' only. Sub-sections are read-only.")

    def j(obj):
        return json.dumps(obj, ensure_ascii=False)

    with conn.cursor() as cur:

        # ── ensure session row exists (session_memory only) ───────────────────
        if schema == "session_memory":
            cur.execute("""
                INSERT INTO session_memory (session_id)
                VALUES (%s) ON CONFLICT (session_id) DO NOTHING;
            """, (session_id,))

        # ══════════════════════════════════════════════════════════════════════
        #  SESSION MEMORY SECTIONS
        # ══════════════════════════════════════════════════════════════════════

        if section == "session_meta":
            if not isinstance(data, dict):
                raise TypeError("session_meta must be a dict")
            cur.execute("""
                UPDATE session_memory
                SET    session_meta = session_meta || %s::jsonb,
                       updated_at  = now()
                WHERE  session_id  = %s;
            """, (j(data), session_id))

        elif section == "persons":
            # upsert by id — handles single dict or list of dicts
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                raise TypeError("persons must be a dict or list of dicts")
            cur.execute("""
                UPDATE session_memory
                SET persons = (
                    WITH
                      existing AS (
                          SELECT value AS p, ordinality - 1 AS idx
                          FROM   jsonb_array_elements(persons) WITH ORDINALITY
                      ),
                      incoming AS (
                          SELECT value AS p FROM jsonb_array_elements(%s::jsonb)
                      ),
                      merged AS (
                          SELECT COALESCE(e.idx, NULL) AS idx,
                                 CASE WHEN e.p IS NOT NULL THEN e.p || i.p ELSE i.p END AS p
                          FROM   incoming i
                          LEFT   JOIN existing e ON e.p->>'id' = i.p->>'id'
                      ),
                      rebuilt AS (
                          SELECT e.idx, COALESCE(m.p, e.p) AS p
                          FROM   existing e LEFT JOIN merged m ON m.idx = e.idx
                          UNION ALL
                          SELECT NULL AS idx, m.p FROM merged m WHERE m.idx IS NULL
                      )
                    SELECT jsonb_agg(p ORDER BY idx NULLS LAST) FROM rebuilt
                ),
                updated_at = now()
                WHERE session_id = %s;
            """, (j(data), session_id))

        elif section in ("group_profile", "joke_tracker"):
            if not isinstance(data, dict):
                raise TypeError(f"{section} must be a dict")
            cur.execute(f"""
                UPDATE session_memory
                SET    {section} = %s::jsonb,
                       updated_at = now()
                WHERE  session_id = %s;
            """, (j(data), session_id))

        elif section == "topics_discussed":
            if isinstance(data, dict):
                cur.execute("""
                    UPDATE session_memory
                    SET    topics_discussed = topics_discussed || %s::jsonb,
                           updated_at      = now()
                    WHERE  session_id      = %s;
                """, (j([data]), session_id))
            elif isinstance(data, list):
                cur.execute("""
                    UPDATE session_memory
                    SET    topics_discussed = %s::jsonb,
                           updated_at      = now()
                    WHERE  session_id      = %s;
                """, (j(data), session_id))
            else:
                raise TypeError("topics_discussed: pass a dict (append) or list (replace all)")

        elif section == "flags":
            if not isinstance(data, dict):
                raise TypeError("flags must be a dict")
            cur.execute("""
                UPDATE session_memory
                SET    flags      = flags || %s::jsonb,
                       updated_at = now()
                WHERE  session_id = %s;
            """, (j(data), session_id))

        elif section == "interaction_log":
            if not isinstance(data, dict):
                raise TypeError("interaction_log entry must be a dict")
            data.setdefault("_written_at", time.time())
            cur.execute("""
                INSERT INTO interaction_log
                    (session_id, seq, ts_elapsed_min, entry_type, intent,
                     response, speaker_id, target, who_reacted_most, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
            """, (
                session_id,
                data.get("seq"),
                data.get("ts_elapsed_min"),
                data.get("type"),
                data.get("intent"),
                data.get("response"),
                data.get("speaker_id"),
                data.get("target"),
                data.get("who_reacted_most"),
                j(data),
            ))

        # ══════════════════════════════════════════════════════════════════════
        #  CURRENT STATE
        # ══════════════════════════════════════════════════════════════════════

        elif section == "snapshot":
            if not isinstance(data, dict):
                raise TypeError("current_state snapshot must be a dict")
            fused       = data.get("fused", {})
            vision      = data.get("vision", {})
            group_audio = data.get("audio", {}).get("group_audio", {})
            cur.execute("""
                INSERT INTO current_state
                    (session_id, ts_ms, elapsed_min,
                     group_energy, group_valence, dominant_quadrant,
                     attention_on_robot, laughter_detected,
                     anyone_distressed, natural_pause_detected,
                     face_count, snapshot)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
            """, (
                session_id,
                data.get("ts_ms"),
                data.get("elapsed_min"),
                fused.get("group_energy"),
                fused.get("group_valence"),
                fused.get("dominant_quadrant"),
                fused.get("attention_on_robot"),
                group_audio.get("laughter_detected", False),
                fused.get("anyone_distressed", False),
                fused.get("natural_pause_detected", False),
                vision.get("face_count"),
                j(data),
            ))

    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  get_from()
# ─────────────────────────────────────────────────────────────────────────────

def get_from(conn, schema: str, section: str,
             query: Optional[dict] = None, session_id: str = "") -> Any:
    """
    Read data from the database. Never modifies anything.

    Parameters
    ----------
    conn       : psycopg2 connection  (from connect())
    schema     : "session_memory"  or  "current_state"
    section    : section name
    query      : optional filter dict (see below)
    session_id : e.g. "ses_abc123"

    session_memory queries
    ----------------------
    "session_meta"
        no query → full dict

    "persons"
        no query              → full list
        {"id": "p1"}          → single person dict or None
        {"quiet_one": True}   → filtered list (any key=value)

    "group_profile"
        no query → full dict

    "joke_tracker"
        no query → full dict

    "topics_discussed"
        no query                    → full list
        {"topic": "promotion"}      → single dict or None
        {"can_revisit": True}       → filtered list

    "flags"
        no query                    → full dict
        {"key": "food_arrived"}     → single flag value

    "interaction_log"
        {"limit": N}                → last N entries        (default 50)
        {"person_id": "p1"}         → entries referencing p1
        {"type": "robot_utterance"}
        {"intent": "joke"}
        {"response": "strong_laugh"}
        {"since_min": 5.0}          → ts_elapsed_min >= value
        (combine freely — all filters are AND)

    current_state queries
    ---------------------
    "snapshot" | "vision" | "audio" | "fused" | "robot" | "table_context"
        no query / no filters       → latest single document
        {"limit": N}                → N most recent (always a list)
        {"since_min": 5.0}          → elapsed_min >= value  (returns list)
        {"since_ts_ms": 1748000000} → ts_ms >= value        (returns list)
        {"laughter": True}          → laughter_detected = true
        {"distressed": True}        → anyone_distressed = true
        {"pause": True}             → natural_pause_detected = true
        {"min_energy": 0.7}         → group_energy >= value
        (combine freely — all filters are AND)
    """

    # ── validation ────────────────────────────────────────────────────────────
    if schema not in ("session_memory", "current_state"):
        raise KeyError(f"Unknown schema '{schema}'. Use 'session_memory' or 'current_state'.")
    if schema == "session_memory" and section not in _SESSION_SECTIONS:
        raise KeyError(f"Unknown section '{section}'. Valid: {sorted(_SESSION_SECTIONS)}")
    if schema == "current_state" and section not in _CURRENT_STATE_SECTIONS:
        raise KeyError(f"Unknown section '{section}'. Valid: {sorted(_CURRENT_STATE_SECTIONS)}")

    q = query or {}

    with conn.cursor() as cur:

        # ══════════════════════════════════════════════════════════════════════
        #  SESSION MEMORY SECTIONS
        # ══════════════════════════════════════════════════════════════════════

        if section == "session_meta":
            cur.execute("SELECT session_meta FROM session_memory WHERE session_id = %s;", (session_id,))
            row = cur.fetchone()
            return row[0] if row else None

        elif section == "persons":
            cur.execute("SELECT persons FROM session_memory WHERE session_id = %s;", (session_id,))
            row = cur.fetchone()
            if not row:
                return []
            persons = row[0] or []
            if not q:
                return persons
            if "id" in q:
                return next((p for p in persons if p.get("id") == q["id"]), None)
            return [p for p in persons if all(p.get(k) == v for k, v in q.items())]

        elif section in ("group_profile", "joke_tracker"):
            cur.execute(f"SELECT {section} FROM session_memory WHERE session_id = %s;", (session_id,))
            row = cur.fetchone()
            return row[0] if row else {}

        elif section == "topics_discussed":
            cur.execute("SELECT topics_discussed FROM session_memory WHERE session_id = %s;", (session_id,))
            row = cur.fetchone()
            if not row:
                return None if "topic" in q else []
            topics = row[0] or []
            if not q:
                return topics
            if "topic" in q:
                return next((t for t in topics if t.get("topic") == q["topic"]), None)
            return [t for t in topics if all(t.get(k) == v for k, v in q.items())]

        elif section == "flags":
            cur.execute("SELECT flags FROM session_memory WHERE session_id = %s;", (session_id,))
            row = cur.fetchone()
            if not row:
                return None if "key" in q else {}
            flags = row[0] or {}
            return flags.get(q["key"]) if "key" in q else flags

        elif section == "interaction_log":
            limit      = int(q.get("limit", 50))
            person_id  = q.get("person_id")
            entry_type = q.get("type")
            intent     = q.get("intent")
            since_min  = q.get("since_min")
            response   = q.get("response")

            conditions = ["session_id = %s"]
            params: list = [session_id]

            if entry_type:
                conditions.append("entry_type = %s"); params.append(entry_type)
            if intent:
                conditions.append("intent = %s"); params.append(intent)
            if response:
                conditions.append("response = %s"); params.append(response)
            if since_min is not None:
                conditions.append("ts_elapsed_min >= %s"); params.append(since_min)
            if person_id:
                conditions.append("""
                    (   speaker_id       = %s
                     OR target           = %s
                     OR who_reacted_most = %s
                     OR data @> %s::jsonb
                    )
                """)
                params += [person_id, person_id, person_id,
                           json.dumps({"per_person_reaction": [{"id": person_id}]})]

            where = " AND ".join(conditions)
            params.append(limit)
            cur.execute(f"""
                SELECT data FROM (
                    SELECT data, ts_elapsed_min, id
                    FROM   interaction_log
                    WHERE  {where}
                    ORDER  BY ts_elapsed_min DESC NULLS FIRST, id DESC
                    LIMIT  %s
                ) sub
                ORDER BY ts_elapsed_min ASC NULLS LAST, id ASC;
            """, params)
            return [row[0] for row in cur.fetchall()]

        # ══════════════════════════════════════════════════════════════════════
        #  CURRENT STATE SECTIONS
        # ══════════════════════════════════════════════════════════════════════

        else:
            _filter_keys = {"since_min", "since_ts_ms", "laughter",
                            "distressed", "pause", "min_energy"}
            has_filters   = bool(_filter_keys & q.keys())
            explicit_limit = "limit" in q
            limit          = int(q.get("limit", 50 if has_filters else 1))
            return_single  = not explicit_limit and not has_filters

            conditions = ["session_id = %s"]
            params: list = [session_id]

            if q.get("since_min") is not None:
                conditions.append("elapsed_min >= %s"); params.append(q["since_min"])
            if q.get("since_ts_ms") is not None:
                conditions.append("ts_ms >= %s"); params.append(q["since_ts_ms"])
            if q.get("laughter") is not None:
                conditions.append("laughter_detected = %s"); params.append(q["laughter"])
            if q.get("distressed") is not None:
                conditions.append("anyone_distressed = %s"); params.append(q["distressed"])
            if q.get("pause") is not None:
                conditions.append("natural_pause_detected = %s"); params.append(q["pause"])
            if q.get("min_energy") is not None:
                conditions.append("group_energy >= %s"); params.append(q["min_energy"])

            select_expr = "snapshot" if section == "snapshot" else f"snapshot->'{section}'"
            where = " AND ".join(conditions)
            params.append(limit)

            cur.execute(f"""
                SELECT {select_expr}
                FROM   current_state
                WHERE  {where}
                ORDER  BY ts_ms DESC
                LIMIT  %s;
            """, params)

            rows = [row[0] for row in cur.fetchall()]
            return (rows[0] if rows else None) if return_single else rows