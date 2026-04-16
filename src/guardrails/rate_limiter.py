"""
Assignment 11 - Rate Limiter Guardrail

This layer throttles abusive traffic with a per-user sliding window.
It blocks requests early so expensive LLM and downstream guardrails
are not invoked for obvious abuse bursts.
"""
from collections import defaultdict, deque
import time

from google.genai import types
from google.adk.plugins import base_plugin
from google.adk.agents.invocation_context import InvocationContext


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed N requests inside a moving time window."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def _resolve_user_id(self, invocation_context: InvocationContext | None) -> str:
        """Extract user ID from callback context with safe fallback."""
        if invocation_context is not None and getattr(invocation_context, "user_id", None):
            return str(invocation_context.user_id)
        return "anonymous"

    def _build_block_message(self, wait_seconds: int) -> types.Content:
        """Return a user-facing message when request is rate limited."""
        return types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=(
                        "Rate limit exceeded. Please wait "
                        f"{wait_seconds} seconds before sending another request."
                    )
                )
            ],
        )

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Apply sliding-window checks before the message reaches the model."""
        del user_message  # Not needed for rate-limit logic.

        self.total_count += 1
        user_id = self._resolve_user_id(invocation_context)
        now = time.time()
        window = self.user_windows[user_id]

        # Remove timestamps that have moved out of the active time window.
        while window and (now - window[0]) > self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            self.blocked_count += 1
            wait_seconds = max(1, int(self.window_seconds - (now - window[0])))
            return self._build_block_message(wait_seconds)

        window.append(now)
        return None
