
"""THESIS AI 3.2 scheduled agent tasks.

Syncs learned knowledge into PostgreSQL and runs
only approved, predefined tasks.
"""

import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "backend")
)

from app.agent32 import db, init_schema


def sync_notes(conn):
    path = ROOT / "knowledge" / "notes.json"

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError(
            "notes.json must be a JSON array"
        )

    count = 0

    for note in data[:450]:
        if not isinstance(note, dict):
            continue

        uid = str(note.get("id", ""))[:120]
        url = str(note.get("url", ""))[:1500]

        if not uid or not url.startswith("https://"):
            continue

        conn.execute(
            """
            INSERT INTO agent_knowledge
            (
                id,
                category,
                title,
                url,
                summary_vi,
                review_status,
                learned_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id)
            DO UPDATE SET
                category = EXCLUDED.category,
                title = EXCLUDED.title,
                url = EXCLUDED.url,
                summary_vi = EXCLUDED.summary_vi,
                review_status = EXCLUDED.review_status,
                learned_at = EXCLUDED.learned_at,
                synced_at = NOW()
            """,
            (
                uid,
                str(
                    note.get("category", "other")
                )[:80],
                str(note.get("title", ""))[:400],
                url,
                str(
                    note.get("summary_vi", "")
                )[:3000],
                str(
                    note.get(
                        "review_status",
                        "needs_review"
                    )
                )[:50],
                str(
                    note.get("learned_at", "")
                )[:100]
            )
        )

        count += 1

    return count


def run_tasks(conn):
    tasks = conn.execute(
        """
        SELECT id, kind
        FROM agent_tasks
        WHERE enabled = TRUE
        """
    ).fetchall()

    for task in tasks:

        # Run each task at most once per UTC day.
        already_ran = conn.execute(
            """
            SELECT 1
            FROM agent_runs
            WHERE task_id = %s
              AND created_at >= date_trunc(
                  'day',
                  now()
              )
            """,
            (task["id"],)
        ).fetchone()

        if already_ran:
            continue

        if task["kind"] == "daily_brief":

            rows = conn.execute(
                """
                SELECT
                    category,
                    title,
                    url,
                    summary_vi,
                    review_status
                FROM agent_knowledge
                ORDER BY synced_at DESC
                LIMIT 5
                """
            ).fetchall()

            result = "\n".join(
                (
                    f"• [{r['category']}] "
                    f"{r['title']}: "
                    f"{r['summary_vi'][:250]} "
                    f"| {r['url']} "
                    f"(review: {r['review_status']})"
                )
                for r in rows
            )

            if not result:
                result = (
                    "Chưa có ghi chú được đồng bộ."
                )

        elif task["kind"] == "memory_review":

            row = conn.execute(
                """
                SELECT count(*) AS n
                FROM agent_rules
                WHERE enabled = TRUE
                """
            ).fetchone()

            result = (
                f"Có {row['n']} quy tắc đang bật. "
                "Xem /agent/rules để quản lý."
            )

        else:
            # Never execute arbitrary commands.
            continue

        conn.execute(
            """
            INSERT INTO agent_runs
            (
                id,
                task_id,
                kind,
                result
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                uuid.uuid4(),
                task["id"],
                task["kind"],
                result[:5000]
            )
        )

        print(
            f"Executed safe task: {task['kind']}"
        )


if __name__ == "__main__":

    init_schema()

    with db() as conn:
        synced = sync_notes(conn)

        print(
            "Knowledge notes synced:",
            synced
        )

        run_tasks(conn)

    print("THESIS agent jobs complete")
