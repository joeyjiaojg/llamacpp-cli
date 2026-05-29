"""Tests for ip_filter.py - IP whitelisting middleware."""

from __future__ import annotations

import ipaddress
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import JSONResponse

from llamacpp_cli.ip_filter import (
    create_ip_filter_middleware,
    is_ip_allowed,
    parse_cidrs,
)


# ---------------------------------------------------------------------------
# parse_cidrs
# ---------------------------------------------------------------------------


class TestParseCidrs:
    def test_single_ipv4_cidr(self):
        nets = parse_cidrs(["10.0.0.0/8"])
        assert len(nets) == 1
        assert isinstance(nets[0], ipaddress.IPv4Network)

    def test_single_ipv6_cidr(self):
        nets = parse_cidrs(["::1/128"])
        assert len(nets) == 1
        assert isinstance(nets[0], ipaddress.IPv6Network)

    def test_multiple_cidrs(self):
        nets = parse_cidrs(["10.0.0.0/8", "192.168.0.0/16"])
        assert len(nets) == 2

    def test_invalid_cidr_is_skipped(self, capsys):
        nets = parse_cidrs(["not-a-cidr"])
        assert nets == []
        captured = capsys.readouterr()
        assert "invalid CIDR" in captured.out

    def test_mix_valid_and_invalid(self, capsys):
        nets = parse_cidrs(["10.0.0.0/8", "bad-cidr", "192.168.1.0/24"])
        assert len(nets) == 2
        captured = capsys.readouterr()
        assert "bad-cidr" in captured.out

    def test_empty_list(self):
        nets = parse_cidrs([])
        assert nets == []

    def test_empty_string_entry_is_skipped(self):
        nets = parse_cidrs(["", "  ", "10.0.0.0/8"])
        assert len(nets) == 1

    def test_host_address_without_prefix(self):
        # plain IPs are treated as host routes
        nets = parse_cidrs(["10.1.2.3"])
        assert len(nets) == 1
        assert nets[0].prefixlen == 32

    def test_ipv6_host_address_without_prefix(self):
        nets = parse_cidrs(["2001:db8::1"])
        assert len(nets) == 1
        assert nets[0].prefixlen == 128


# ---------------------------------------------------------------------------
# is_ip_allowed
# ---------------------------------------------------------------------------


class TestIsIpAllowed:
    def _nets(self, cidrs: list[str]):
        return parse_cidrs(cidrs)

    def test_exact_match_in_subnet(self):
        nets = self._nets(["10.0.0.0/8"])
        assert is_ip_allowed("10.1.2.3", nets) is True

    def test_ip_outside_subnet(self):
        nets = self._nets(["10.0.0.0/8"])
        assert is_ip_allowed("192.168.1.1", nets) is False

    def test_boundary_ip_in_range(self):
        nets = self._nets(["192.168.1.0/24"])
        assert is_ip_allowed("192.168.1.255", nets) is True

    def test_boundary_ip_outside_range(self):
        nets = self._nets(["192.168.1.0/24"])
        assert is_ip_allowed("192.168.2.0", nets) is False

    def test_multiple_networks_first_matches(self):
        nets = self._nets(["10.0.0.0/8", "192.168.0.0/16"])
        assert is_ip_allowed("10.50.0.1", nets) is True

    def test_multiple_networks_second_matches(self):
        nets = self._nets(["10.0.0.0/8", "192.168.0.0/16"])
        assert is_ip_allowed("192.168.1.1", nets) is True

    def test_multiple_networks_none_matches(self):
        nets = self._nets(["10.0.0.0/8", "192.168.0.0/16"])
        assert is_ip_allowed("172.16.0.1", nets) is False

    def test_empty_networks_denies_all(self):
        assert is_ip_allowed("10.0.0.1", []) is False

    def test_invalid_ip_string(self):
        nets = self._nets(["10.0.0.0/8"])
        assert is_ip_allowed("not-an-ip", nets) is False

    def test_ipv6_loopback_allowed(self):
        nets = self._nets(["::1/128"])
        assert is_ip_allowed("::1", nets) is True

    def test_ipv6_loopback_not_in_ipv4_network(self):
        nets = self._nets(["10.0.0.0/8"])
        assert is_ip_allowed("::1", nets) is False

    def test_ipv6_range(self):
        nets = self._nets(["2001:db8::/32"])
        assert is_ip_allowed("2001:db8::1", nets) is True
        assert is_ip_allowed("2001:db9::1", nets) is False


# ---------------------------------------------------------------------------
# create_ip_filter_middleware  (integration-style)
# ---------------------------------------------------------------------------


def _make_request(ip: str, forwarded_for: str | None = None) -> MagicMock:
    """Build a minimal mock Request object."""
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = ip
    headers: dict[str, str] = {}
    if forwarded_for is not None:
        headers["x-forwarded-for"] = forwarded_for
    req.headers = headers
    return req


@pytest.mark.asyncio
class TestIpFilterMiddleware:
    async def test_allowed_ip_passes_through(self):
        middleware = create_ip_filter_middleware(["10.0.0.0/8"])
        call_next = AsyncMock(return_value=MagicMock(status_code=200))
        req = _make_request("10.1.2.3")
        resp = await middleware(req, call_next)
        call_next.assert_awaited_once_with(req)
        assert resp.status_code == 200

    async def test_blocked_ip_returns_403(self):
        middleware = create_ip_filter_middleware(["10.0.0.0/8"])
        call_next = AsyncMock()
        req = _make_request("172.16.0.1")
        resp = await middleware(req, call_next)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 403
        call_next.assert_not_awaited()

    async def test_multiple_cidrs(self):
        middleware = create_ip_filter_middleware(["10.0.0.0/8", "192.168.0.0/16"])
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        req_ok_1 = _make_request("10.5.6.7")
        await middleware(req_ok_1, call_next)
        assert call_next.await_count == 1

        req_ok_2 = _make_request("192.168.1.100")
        await middleware(req_ok_2, call_next)
        assert call_next.await_count == 2

        req_blocked = _make_request("8.8.8.8")
        resp = await middleware(req_blocked, call_next)
        assert resp.status_code == 403
        assert call_next.await_count == 2

    async def test_x_forwarded_for_used_when_present(self):
        middleware = create_ip_filter_middleware(["10.0.0.0/8"])
        call_next = AsyncMock(return_value=MagicMock(status_code=200))
        # direct connection from 1.2.3.4 but real client is 10.0.0.1
        req = _make_request("1.2.3.4", forwarded_for="10.0.0.1, 1.2.3.4")
        resp = await middleware(req, call_next)
        call_next.assert_awaited_once()
        assert resp.status_code == 200

    async def test_no_client_returns_403(self):
        middleware = create_ip_filter_middleware(["10.0.0.0/8"])
        call_next = AsyncMock()
        req = MagicMock()
        req.client = None
        req.headers = {}
        resp = await middleware(req, call_next)
        assert resp.status_code == 403

    async def test_empty_cidr_list_blocks_all(self):
        middleware = create_ip_filter_middleware([])
        call_next = AsyncMock()
        req = _make_request("10.0.0.1")
        resp = await middleware(req, call_next)
        assert resp.status_code == 403

    async def test_invalid_cidr_skipped_and_blocks_all(self, capsys):
        middleware = create_ip_filter_middleware(["not-valid"])
        call_next = AsyncMock()
        req = _make_request("10.0.0.1")
        resp = await middleware(req, call_next)
        assert resp.status_code == 403

    async def test_ipv6_allowed(self):
        middleware = create_ip_filter_middleware(["::1/128"])
        call_next = AsyncMock(return_value=MagicMock(status_code=200))
        req = _make_request("::1")
        resp = await middleware(req, call_next)
        call_next.assert_awaited_once()
        assert resp.status_code == 200

    async def test_ipv6_blocked(self):
        middleware = create_ip_filter_middleware(["::1/128"])
        call_next = AsyncMock()
        req = _make_request("2001:db8::1")
        resp = await middleware(req, call_next)
        assert resp.status_code == 403

    async def test_response_body_contains_ip(self):
        import json

        middleware = create_ip_filter_middleware(["10.0.0.0/8"])
        call_next = AsyncMock()
        req = _make_request("1.2.3.4")
        resp = await middleware(req, call_next)
        body = json.loads(resp.body)
        assert body["ip"] == "1.2.3.4"
        assert body["error"] == "IP not allowed"
