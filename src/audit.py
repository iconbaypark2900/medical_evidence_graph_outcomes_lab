"""Append-only audit log.

The README promises "full audit trails of ingested datasets,
transformations, and queries" and there was no audit code at all. This
became both more tractable and more conspicuous once API keys landed:
there is now an identity to attribute an action to.

What is recorded is metadata -- who, when, which action, over how many
patients, which outcomes, which model version. What is deliberately not
recorded is any patient content. An audit log that accumulates the data it
is auditing becomes the largest copy of that data in the system, held in
the file least likely to be access-controlled.

Actors are stored as a short hash of the API key. That is enough to say
"the same caller did both of these" and to match a key you hold against
the log, and it means the log is not a list of live credentials.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

DEFAULT_AUDIT_PATH = Path(os.environ.get("MEG_AUDIT_LOG", ".audit/audit.jsonl"))

# Field names that carry patient collections on this API.
FORBIDDEN_DETAIL_KEYS = frozenset({
    "patients", "patient_data", "training_data", "patient_cohort",
    "comparator_cohort",
})


def _carries_records(value: Any) -> bool:
    """Whether a value looks like structured records rather than metadata.

    Checked by shape as well as by name: the point is to stop a payload
    reaching the log, and the key it arrives under is the caller's choice.
    A count named `results` is fine; a list of patient dicts is not,
    whatever it is called.
    """
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, (dict, list, tuple)) for item in value)
    return False


def actor_id(api_key: Optional[str]) -> str:
    """Stable, non-reversible identifier for a caller."""
    if not api_key:
        return "anonymous"
    return "key:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


class AuditLog:
    """JSONL, one event per line, append-only.

    A flat file rather than a table: an audit record's value comes from
    being append-only and easy to ship elsewhere, and a format that
    survives the database being unavailable is worth more here than one
    that can be queried in place.
    """

    def __init__(self, path: Path = DEFAULT_AUDIT_PATH):
        self.path = Path(path)
        # Writes are serialised so concurrent requests cannot interleave
        # halves of two records into one corrupt line.
        self._lock = threading.Lock()

    def record(self, action: str, actor: str = "anonymous",
               outcome: str = "success", **details: Any) -> Dict[str, Any]:
        rejected = sorted(
            set(FORBIDDEN_DETAIL_KEYS & set(details))
            | {k for k, v in details.items() if _carries_records(v)})
        if rejected:
            # An audit log that accumulates the data it audits becomes the
            # largest copy of that data in the system, in the file least
            # likely to be access-controlled.
            raise ValueError(
                f"Refusing to audit {rejected}: the audit log records "
                f"metadata, not the data being audited.")

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "outcome": outcome,
            **details,
        }

        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, default=str) + "\n")
        except OSError as e:
            # An unwritable audit log must not take the API down, but it
            # must not pass unnoticed either.
            logger.error(f"Could not write audit record to {self.path}: {e}")

        return event

    def read(self, limit: int = 100, action: Optional[str] = None,
             actor: Optional[str] = None) -> List[Dict[str, Any]]:
        """Most recent events first."""
        if not self.path.exists():
            return []

        events = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # One damaged line must not hide the rest of the log.
                    logger.warning("Skipping unparseable audit line")
        except OSError as e:
            logger.error(f"Could not read audit log {self.path}: {e}")
            return []

        if action:
            events = [e for e in events if e.get("action") == action]
        if actor:
            events = [e for e in events if e.get("actor") == actor]
        return list(reversed(events))[:limit]
