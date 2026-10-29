"""Claude (Anthropic API) as an LLM backend for Claude mode.

Same interface as the local Ollama backend (``stream_chat`` yields text), so the
assistant can speak Claude's replies sentence by sentence. Spoken replies are
short, so effort is ``low``; Claude Opus 5.5 always thinks adaptively and
``max_tokens`` leaves room for that. Server-side refusal fallbacks are on: a
request a policy check declines is retried on a fallback model instead of
simply stopping.

Every request records its token usage in ``last_usage`` for the audit log.
The API key comes from the system keyring (``orion`` / ``anthropic-api-key``)
or ANTHROPIC_API_KEY, and is never logged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterator

from brain.llm import LLMError, Message

log = logging.getLogger(__name__)

KEYRING_SERVICE = "orion"
KEYRING_NAME = "anthropic-api-key"

# USD per million tokens (input, output), for the audit log's cost estimate.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def api_key() -> str | None:
    """The key, or None if it's missing OR empty (an empty saved value counts as no key)."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key:
        return key
    try:
        import keyring

        return (keyring.get_password(KEYRING_SERVICE, KEYRING_NAME) or "").strip() or None
    except Exception:
        return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, PRICES["claude-opus-5-5"])
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


class ClaudeBackend:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 4096,
                 fallbacks: bool = True, timeout_s: float = 60.0, client: Any = None) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.timeout_s = timeout_s
        self._client = client
        self.last_usage: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, cloud: dict[str, Any]) -> "ClaudeBackend":
        return cls(model=cloud.get("model", "claude-opus-5-5"), effort=cloud.get("effort", "low"),
                   max_tokens=cloud.get("max_tokens", 4096), fallbacks=cloud.get("fallbacks", True),
                   timeout_s=cloud.get("timeout_s", 60))

    def _get_client(self):
        if self._client is None:
            import anthropic

            key = api_key()
            if not key:
                raise LLMError("I don't have a Claude API key. Add one with: python main.py secrets set anthropic.")
            self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s, max_retries=1)
        return self._client

    def warm_up(self) -> None:
        pass   # no local model to load; checking the key would cost a request

    def request_body(self, messages: list[Message]) -> dict[str, Any]:
        """The exact request sent (minus credentials), also used for the audit log."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                                "messages": convo, "output_config": {"effort": self.effort}}
        if self.fallbacks:
            body["betas"] = ["server-side-fallback-2026-07-01"]
            body["fallbacks"] = "default"
        return body

    def stream_chat(self, messages: list[Message], tools: list[dict] | None = None) -> Iterator[str]:
        import anthropic

        body = self.request_body(messages)
        if tools:
            body["tools"] = tools
        self.last_usage = None
        try:
            client = self._get_client()
            with client.beta.messages.stream(**body) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
        except LLMError:
            raise
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude rejected my API key. Check it in the Claude Console.", str(exc)) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("My Claude API key isn't allowed to do that.", str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude says I've hit the rate or spending limit.", str(exc)) from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Claude doesn't recognise the model {self.model}.", str(exc)) from exc
        except anthropic.BadRequestError as exc:
            raise LLMError("Claude couldn't accept that request.", str(exc)) from exc
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise LLMError("I can't reach Claude right now. Is the internet working?", str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError("Claude's service is having trouble right now.", str(exc)) from exc

        usage = getattr(final, "usage", None)
        tin = getattr(usage, "input_tokens", 0) or 0
        tout = getattr(usage, "output_tokens", 0) or 0
        self.last_usage = {"model": getattr(final, "model", self.model), "input_tokens": tin,
                           "output_tokens": tout, "stop_reason": final.stop_reason,
                           "cost_usd": round(estimate_cost(self.model, tin, tout), 5)}
        if final.stop_reason == "refusal":
            yield " I'm not able to help with that one."
        elif final.stop_reason == "max_tokens" and not any(
                getattr(b, "type", "") == "text" and b.text for b in final.content):
            raise LLMError("Claude ran out of room before answering.")
