import pytest

from core.net_guard import NetworkPolicyError, require_loopback_url


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"])
def test_loopback_allowed(url):
    require_loopback_url(url)


@pytest.mark.parametrize("url", ["http://0.0.0.0:11434", "http://192.168.1.5:11434", "http://example.com"])
def test_non_loopback_rejected(url):
    with pytest.raises(NetworkPolicyError):
        require_loopback_url(url)
