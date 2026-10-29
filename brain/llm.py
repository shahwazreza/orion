"""Language-model backends behind one small interface.

Phase 1 ships the local Ollama backend. Phase 2 adds an Anthropic backend
behind the same interface, selected by the router.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Protocol

import httpx

from core.net_guard import require_loopback_url

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """A backend failure with a message that is safe to say out loud."""

    def __init__(self, spoken: str, detail: str = "") -> None:
        super().__init__(detail or spoken)
        self.spoken = spoken


class LLMBackend(Protocol):
    name: str

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield reply text incrementally."""
        ...

    def warm_up(self) -> None:
        ...


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        temperature: float = 0.6,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        think: bool | None = None,
        timeout_s: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        require_loopback_url(host)
        self.host = host.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.think = think
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        # Connect fast (local socket); allow a long read for slow CPU generation.
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=3.0), trust_env=False
        )

    @classmethod
    def from_config(cls, llm: dict[str, Any]) -> "OllamaBackend":
        return cls(
            host=llm["host"],
            model=llm["model"],
            temperature=llm.get("temperature", 0.6),
            num_ctx=llm.get("num_ctx", 4096),
            keep_alive=str(llm.get("keep_alive", "30m")),
            think=llm.get("think"),
            timeout_s=llm.get("timeout_s", 90),
        )

    def _payload(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        if self.think is not None:
            body["think"] = self.think
        return body

    def warm_up(self) -> None:
        """Load the model into memory so the first real request isn't slow."""
        try:
            r = self._client.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "keep_alive": self.keep_alive},
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._translate(exc) from exc

    def loaded_models(self) -> list[dict[str, Any]]:
        """What Ollama has in memory: name, total size and VRAM share in MB."""
        try:
            r = self._client.get(f"{self.host}/api/ps")
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        return [
            {"name": m.get("name"), "size_mb": round(m.get("size", 0) / 2**20),
             "vram_mb": round(m.get("size_vram", 0) / 2**20)}
            for m in r.json().get("models", [])
        ]

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        try:
            with self._client.stream(
                "POST", f"{self.host}/api/chat", json=self._payload(messages, True)
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise httpx.HTTPStatusError(resp.text, request=resp.request, response=resp)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise LLMError("My local model reported an error.", chunk["error"])
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise self._translate(exc) from exc

    def _translate(self, exc: httpx.HTTPError) -> LLMError:
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return LLMError("I can't reach Ollama. Is it running?", str(exc))
        if isinstance(exc, httpx.ReadTimeout):
            return LLMError("My local model took too long to answer.", str(exc))
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 404:
                return LLMError(
                    f"The model {self.model} isn't installed in Ollama.", exc.response.text
                )
            return LLMError("My local model returned an error.", exc.response.text)
        return LLMError("Something went wrong talking to my local model.", str(exc))


class Conversation:
    """Short rolling chat history, kept in memory only."""

    def __init__(self, system_prompt: str, max_turns: int = 8) -> None:
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.examples: list[Message] = []   # few-shot turns, shown before the real history
        self.context: str = ""              # per-turn extra system context (memories, past talk)
        self._turns: list[Message] = []

    def messages_with(self, user_text: str) -> list[Message]:
        system = self.system_prompt + (f"\n{self.context}" if self.context else "")
        return [{"role": "system", "content": system}, *self.examples, *self._turns,
                {"role": "user", "content": user_text}]

    def commit(self, user_text: str, reply: str) -> None:
        self._turns += [{"role": "user", "content": user_text},
                        {"role": "assistant", "content": reply}]
        excess = len(self._turns) - 2 * self.max_turns
        if excess > 0:
            del self._turns[:excess]

    def clear(self) -> None:
        self._turns.clear()


def make_llm(llm_cfg: dict[str, Any]) -> LLMBackend:
    backend = llm_cfg.get("backend", "ollama")
    if backend == "ollama":
        return OllamaBackend.from_config(llm_cfg)
    raise ValueError(f"Unknown LLM backend '{backend}'")
