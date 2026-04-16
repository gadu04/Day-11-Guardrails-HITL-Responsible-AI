"""
Assignment 11 - Audit Logging and Monitoring

AuditLogPlugin captures every interaction (input/output/latency/block source)
and exports JSON for compliance. MonitoringAlert computes operational metrics
and emits alerts when risky trends exceed thresholds.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import time

from google.genai import types
from google.adk.plugins import base_plugin
from google.adk.agents.invocation_context import InvocationContext


def _extract_text(content: types.Content | None) -> str:
    """Flatten ADK Content objects into a plain string for logging."""
    if not content or not content.parts:
        return ""

    text = ""
    for part in content.parts:
        if hasattr(part, "text") and part.text:
            text += part.text
    return text


def detect_block_layer(response_text: str) -> str | None:
    """Map response text to the layer that likely blocked or transformed it."""
    normalized = response_text.lower()
    if "rate limit exceeded" in normalized:
        return "rate_limiter"
    if "cannot process requests that attempt" in normalized:
        return "input_guardrail_injection"
    if "i can only help with banking topics" in normalized:
        return "input_guardrail_topic"
    if "message is empty" in normalized or "message is too long" in normalized:
        return "input_guardrail_validation"
    if "i cannot provide that response" in normalized:
        return "output_guardrail_judge"
    if "[redacted]" in normalized:
        return "output_guardrail_redaction"
    return None


class AuditLogPlugin(base_plugin.BasePlugin):
    """Persist request/response metadata for traceability and incident analysis."""

    def __init__(self):
        super().__init__(name="audit_log")
        self.logs: list[dict] = []
        self.total_count = 0
        self._pending_indices: deque[int] = deque()

    def record_interaction(
        self,
        *,
        user_id: str,
        input_text: str,
        output_text: str,
        latency_ms: float,
    ):
        """Record a completed interaction explicitly (runner-side fallback path)."""
        blocked_layer = detect_block_layer(output_text)
        self.logs.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "input": input_text,
                "output": output_text,
                "latency_ms": round(latency_ms, 2),
                "blocked": blocked_layer is not None and blocked_layer != "output_guardrail_redaction",
                "blocked_layer": blocked_layer,
                "judge_failed": blocked_layer == "output_guardrail_judge",
                "redacted": blocked_layer == "output_guardrail_redaction",
            }
        )
        self.total_count += 1

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Capture input event and start time. This layer never blocks."""
        user_id = "anonymous"
        if invocation_context is not None and getattr(invocation_context, "user_id", None):
            user_id = str(invocation_context.user_id)

        now = time.time()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "input": _extract_text(user_message),
            "output": None,
            "latency_ms": None,
            "blocked": False,
            "blocked_layer": None,
            "judge_failed": False,
            "redacted": False,
            "_start_time": now,
        }
        self.logs.append(record)
        self._pending_indices.append(len(self.logs) - 1)
        self.total_count += 1
        return None

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Capture output and latency after generation. This layer never mutates."""
        del callback_context

        if not self._pending_indices:
            return llm_response

        record_index = self._pending_indices.popleft()
        record = self.logs[record_index]
        response_text = ""
        if hasattr(llm_response, "content"):
            response_text = _extract_text(llm_response.content)

        end_time = time.time()
        blocked_layer = detect_block_layer(response_text)

        record["output"] = response_text
        record["latency_ms"] = round((end_time - record["_start_time"]) * 1000, 2)
        record["blocked"] = blocked_layer is not None and blocked_layer != "output_guardrail_redaction"
        record["blocked_layer"] = blocked_layer
        record["judge_failed"] = blocked_layer == "output_guardrail_judge"
        record["redacted"] = blocked_layer == "output_guardrail_redaction"
        record.pop("_start_time", None)
        return llm_response

    def export_json(self, filepath: str = "security_audit.json"):
        """Write audit records to JSON for assignment submission evidence."""
        safe_records = []
        for record in self.logs:
            copy = dict(record)
            copy.pop("_start_time", None)
            safe_records.append(copy)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(safe_records, file, indent=2, ensure_ascii=False)


class MonitoringAlert:
    """Aggregate guardrail metrics and raise threshold-based alerts."""

    def __init__(
        self,
        *,
        rate_limiter,
        input_guardrail,
        output_guardrail,
        audit_log,
        block_rate_threshold: float = 0.45,
        rate_limit_threshold: int = 5,
        judge_fail_threshold: float = 0.20,
    ):
        self.rate_limiter = rate_limiter
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.audit_log = audit_log
        self.block_rate_threshold = block_rate_threshold
        self.rate_limit_threshold = rate_limit_threshold
        self.judge_fail_threshold = judge_fail_threshold

    def metrics(self) -> dict:
        """Compute summary metrics used by production monitoring dashboards."""
        total = max(1, self.audit_log.total_count)
        total_blocks = (
            self.rate_limiter.blocked_count
            + self.input_guardrail.blocked_count
            + self.output_guardrail.blocked_count
        )
        output_total = max(1, self.output_guardrail.total_count)

        return {
            "total_requests": self.audit_log.total_count,
            "total_blocks": total_blocks,
            "overall_block_rate": total_blocks / total,
            "rate_limit_hits": self.rate_limiter.blocked_count,
            "input_blocks": self.input_guardrail.blocked_count,
            "judge_fail_blocks": self.output_guardrail.blocked_count,
            "redactions": self.output_guardrail.redacted_count,
            "judge_fail_rate": self.output_guardrail.blocked_count / output_total,
        }

    def check_metrics(self) -> list[str]:
        """Return alert messages when metrics exceed configured thresholds."""
        data = self.metrics()
        alerts = []

        if data["overall_block_rate"] > self.block_rate_threshold:
            alerts.append(
                "ALERT: Overall block rate is high "
                f"({data['overall_block_rate']:.1%})."
            )

        if data["rate_limit_hits"] > self.rate_limit_threshold:
            alerts.append(
                "ALERT: Rate-limit hits exceed threshold "
                f"({data['rate_limit_hits']})."
            )

        if data["judge_fail_rate"] > self.judge_fail_threshold:
            alerts.append(
                "ALERT: Judge fail rate is elevated "
                f"({data['judge_fail_rate']:.1%})."
            )

        return alerts
