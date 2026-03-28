from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _project_root() -> Path:
    # utils/error_utils.py -> FactorGenAgent/
    return Path(__file__).resolve().parents[1]


def _get_error_events_path() -> Path:
    return _project_root() / "error_events.jsonl"


def record_error_event(
    stage: str,
    error: Any,
    current_output: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Append one error event to unified logs/error_events.jsonl.
    Returns the event dict for immediate reuse.
    """
    event: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "error": str(error),
    }
    if current_output is not None:
        event["current_output"] = (
            current_output if isinstance(current_output, (str, int, float, bool, list, dict)) else repr(current_output)
        )
    if extra:
        event["extra"] = extra

    log_path = _get_error_events_path()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def build_retry_feedback(event: Dict[str, Any]) -> str:
    """
    Build retry context text that can be appended to LLM user_content.
    """
    lines = [
        "[RETRY FEEDBACK - STRICT FIX REQUIRED]",
        f"Stage: {event.get('stage')}",
        f"Error: {event.get('error')}",
    ]
    output = event.get("current_output")
    if output is not None:
        lines.extend(
            [
                "Previous output:",
                str(output),
            ]
        )
    extra = event.get("extra")
    if extra:
        lines.extend(
            [
                "Extra context:",
                str(extra),
            ]
        )
    lines.append("Now output ONLY valid JSON in the required schema. Do not include markdown fences.")
    return "\n".join(lines)

