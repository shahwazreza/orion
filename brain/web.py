"""Web search and page reading: the only two things Orion's search loop can do.

- ``web_search(query)``: titles, URLs and snippets from SearXNG (local container
  on 127.0.0.1) or the Brave Search API. Results are cached briefly.
- ``read_page(url)``: fetch a public web page and extract its main text with
  trafilatura, trimmed to fit the local model's context.

Safety: pages are DATA. read_page refuses anything that isn't a public internet
address (no localhost, private networks, or link-local), so a page can't point
Orion at Ollama, SearXNG, or your router. Responses are size-capped and only
HTML or plain text is read.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import httpx

from core.net_guard import require_loopback_url

log = logging.getLogger(__name__)

MAX_PAGE_BYTES = 2_000_000
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) Orion/0.2 (personal assistant)"


class SearchError(RuntimeError):
    pass


@dataclass
class Result:
    title: str
    url: str
    snippet: str

    @property
    def site(self) -> str:
        host = urlparse(self.url).hostname or ""
        return host.removeprefix("www.")


class SearchBackend(Protocol):
    name: str

    def search(self, query: str, n: int = 6) -> list[Result]: ...


class SearxngBackend:
    name = "searxng"

    def __init__(self, url: str = "http://127.0.0.1:8888", timeout_s: float = 15.0,
                 client: httpx.Client | None = None) -> None:
        require_loopback_url(url)        # the local container only
        self.url = url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s, trust_env=False)

    def search(self, query: str, n: int = 6) -> list[Result]:
        try:
            r = self._client.get(f"{self.url}/search", params={"q": query, "format": "json"})
            r.raise_for_status()
            rows = r.json().get("results", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError("My local search engine isn't responding. Is SearXNG running?") from exc
        return [Result(x.get("title", ""), x.get("url", ""), x.get("content", "")) for x in rows[:n]
                if x.get("url", "").startswith(("http://", "https://"))]


class BraveBackend:
    name = "brave"

    def __init__(self, api_key: str, timeout_s: float = 15.0, client: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=timeout_s)

    def search(self, query: str, n: int = 6) -> list[Result]:
        try:
            r = self._client.get("https://api.search.brave.com/res/v1/web/search",
                                 params={"q": query, "count": n},
                                 headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"})
            r.raise_for_status()
            rows = r.json().get("web", {}).get("results", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError("Brave Search isn't responding.") from exc
        return [Result(x.get("title", ""), x.get("url", ""), x.get("description", "")) for x in rows[:n]]


class CachedSearch:
    def __init__(self, backend: SearchBackend, ttl_s: float = 900.0) -> None:
        self.backend = backend
        self.ttl_s = ttl_s
        self._cache: dict[str, tuple[float, list[Result]]] = {}

    @property
    def name(self) -> str:
        return self.backend.name

    def search(self, query: str, n: int = 6) -> list[Result]:
        key = " ".join(query.lower().split())
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl_s:
            return hit[1][:n]
        results = self.backend.search(query, n)
        self._cache[key] = (time.monotonic(), results)
        return results


def is_public_url(url: str) -> bool:
    """Only http(s) to a host that resolves to public internet addresses."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            return False
    return True


class PageReader:
    def __init__(self, timeout_s: float = 10.0, max_chars: int = 2500,
                 client: httpx.Client | None = None, url_check=is_public_url) -> None:
        self.max_chars = max_chars
        self.url_check = url_check
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False,
                                              headers={"User-Agent": USER_AGENT})

    def read(self, url: str, max_redirects: int = 3) -> str | None:
        """Main text of a public page, trimmed; None if it can't or mustn't be read."""
        for _ in range(max_redirects + 1):
            if not self.url_check(url):
                log.info("Refused to fetch non-public URL")
                return None
            try:
                with self._client.stream("GET", url) as r:
                    if r.is_redirect and r.headers.get("location"):
                        url = str(r.url.join(r.headers["location"]))
                        continue          # re-check the redirect target too
                    ctype = r.headers.get("content-type", "")
                    if r.status_code != 200 or not ctype.startswith(("text/html", "text/plain",
                                                                     "application/xhtml")):
                        return None
                    body = b""
                    for chunk in r.iter_bytes():
                        body += chunk
                        if len(body) > MAX_PAGE_BYTES:
                            break
            except httpx.HTTPError:
                return None
            return self._extract(body.decode(r.encoding or "utf-8", errors="replace"), ctype)
        return None

    def _extract(self, html: str, ctype: str) -> str | None:
        if ctype.startswith("text/plain"):
            text = html
        else:
            import trafilatura

            text = trafilatura.extract(html, include_comments=False, include_tables=False,
                                       favor_precision=True) or ""
        text = " ".join(text.split())
        return text[: self.max_chars] if text else None
