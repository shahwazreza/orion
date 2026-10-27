"""Network sanity checks for the dorm rule: Orion never listens, only dials out."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import psutil


class NetworkPolicyError(RuntimeError):
    pass


def require_loopback_url(url: str) -> None:
    """Raise unless ``url`` points at this machine's loopback interface."""
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise NetworkPolicyError(f"Local model host must be loopback, got '{host}'")


def exposed_listeners(port: int) -> list[str]:
    """Addresses (non-loopback) on which something listens on ``port``.

    Used to warn if Ollama was started with OLLAMA_HOST=0.0.0.0, which would
    expose it to the dorm network.
    """
    found = []
    try:
        conns = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, OSError):
        return found
    for c in conns:
        if c.status != psutil.CONN_LISTEN or not c.laddr or c.laddr.port != port:
            continue
        try:
            if not ipaddress.ip_address(c.laddr.ip).is_loopback:
                found.append(c.laddr.ip)
        except ValueError:
            found.append(str(c.laddr.ip))
    return found


def port_of(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port:
        return parsed.port
    return socket.getservbyname(parsed.scheme or "http")
