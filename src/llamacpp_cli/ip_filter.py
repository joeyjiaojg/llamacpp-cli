"""IP filtering middleware for lb-proxy."""

from __future__ import annotations

import ipaddress

from fastapi import Request
from fastapi.responses import JSONResponse


def parse_cidrs(allowed_cidrs: list[str]) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse a list of CIDR strings into network objects, skipping invalid entries."""
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for cidr in allowed_cidrs:
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            print(f"Warning: invalid CIDR {cidr!r}, skipping")
    return networks


def is_ip_allowed(
    client_ip: str,
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    """Return True if client_ip falls within any of the allowed networks.

    An empty network list means no IPs are allowed (deny-all by default when
    IP filtering is enabled but no valid CIDRs were provided).
    """
    try:
        ip = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return any(ip in network for network in networks)


def create_ip_filter_middleware(allowed_cidrs: list[str]):
    """Create middleware that allows only whitelisted IPs.

    Args:
        allowed_cidrs: List of CIDR strings (e.g. ["10.0.0.0/8", "::1/128"]).
                       Plain IP addresses (no prefix) are also accepted and
                       treated as a host route (/32 for IPv4, /128 for IPv6).

    Returns:
        An async middleware callable compatible with FastAPI / Starlette.
    """
    networks = parse_cidrs(allowed_cidrs)

    async def ip_filter_middleware(request: Request, call_next):
        # Prefer X-Forwarded-For when running behind a trusted reverse proxy,
        # but fall back to the direct connection IP.
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the leftmost (originating) address
            client_ip = forwarded_for.split(",")[0].strip()
        elif request.client is not None:
            client_ip = request.client.host
        else:
            client_ip = ""

        if is_ip_allowed(client_ip, networks):
            return await call_next(request)

        return JSONResponse(
            status_code=403,
            content={"error": "IP not allowed", "ip": client_ip},
        )

    return ip_filter_middleware
