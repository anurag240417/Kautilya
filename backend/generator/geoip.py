"""Offline GeoIP resolver for country and ASN lookup.

Provides IP → country/ASN resolution using only local data.
No runtime network calls are made.

Two resolution strategies are supported:

1. **Node pool lookup** (primary): Uses the node pool's IP assignments
   as the authoritative mapping for all synthetic IPs. This is always
   consistent with the generated data.

2. **Built-in IP range fallback**: For IPs not in the node pool, uses
   a deterministic mapping based on IANA regional registry allocations.
   This covers the case where the API or frontend encounters IPs
   outside the generator's pool.

See CONTEXT.md §3 and AGENTS.md §10 for offline requirements.
"""

from dataclasses import dataclass

from backend.generator.node_pool import NetworkNode


@dataclass(frozen=True)
class GeoIPResult:
    """Result of a GeoIP lookup.

    Attributes:
        country: ISO 3166-1 alpha-2 country code, or None if unknown.
        asn: Autonomous System Number, or None if unknown.
        source: How this result was resolved ('node_pool' or 'builtin_range').
    """

    country: str | None
    asn: int | None
    source: str


# --- Built-in IP range → country/ASN mapping ---
# Based on simplified IANA regional allocations.
# Each entry: (first_octet_start, first_octet_end, country, asn)
# This is intentionally coarse — it's a fallback, not a replacement
# for a full GeoIP database.

_BUILTIN_RANGES: list[tuple[int, int, str, int]] = [
    # North America (ARIN)
    (3, 4, "US", 15169),
    (6, 7, "US", 7922),
    (8, 9, "US", 3356),
    (11, 12, "US", 20940),
    (13, 15, "US", 14618),
    (16, 19, "US", 16509),
    (20, 23, "US", 7018),
    (24, 24, "CA", 577),
    (25, 30, "US", 2914),
    (31, 31, "US", 209),
    (32, 35, "US", 6939),
    (38, 39, "US", 174),
    (40, 44, "US", 3356),
    (45, 47, "JP", 2497),
    (48, 51, "US", 701),
    (52, 54, "MX", 8151),
    (55, 57, "BR", 28573),
    (58, 60, "JP", 17676),
    (61, 61, "AU", 4764),
    (62, 62, "NL", 60781),
    (63, 66, "US", 6461),
    (67, 68, "US", 11351),
    (69, 72, "US", 22394),
    (73, 76, "US", 7922),
    # Europe (RIPE)
    (77, 79, "DE", 24940),
    (80, 82, "GB", 5089),
    (83, 85, "FR", 16276),
    (86, 88, "DE", 3320),
    (89, 91, "RU", 12389),
    (92, 94, "SE", 29518),
    (95, 95, "NL", 20857),
    # Asia-Pacific (APNIC)
    (101, 103, "IN", 9498),
    (104, 108, "US", 13335),
    (110, 112, "KR", 4766),
    (113, 115, "JP", 4713),
    (116, 118, "CN", 4134),
    (119, 121, "KR", 9318),
    (122, 125, "JP", 2497),
    (126, 126, "JP", 9370),
    # More Europe
    (128, 130, "DE", 51167),
    (131, 133, "FR", 12876),
    (134, 137, "IT", 12874),
    (138, 140, "GB", 2856),
    (141, 143, "DE", 6724),
    (144, 147, "NL", 1101),
    (148, 150, "CH", 13030),
    (151, 155, "ES", 12715),
    (156, 159, "PL", 5617),
    (160, 163, "AT", 8447),
    (164, 167, "CA", 812),
    (168, 170, "BR", 8167),
    (171, 175, "SG", 4657),
    (176, 178, "RU", 31133),
    (179, 179, "BR", 7738),
    (180, 183, "AU", 1221),
    (184, 187, "US", 46606),
    (188, 191, "FI", 24940),
    (193, 195, "CZ", 25248),
    (196, 197, "ZA", 3741),
    (199, 199, "US", 33070),
    (200, 201, "BR", 16735),
    (202, 203, "AU", 7545),
    (204, 209, "US", 11404),
    (210, 211, "KR", 3786),
    (212, 213, "UA", 13188),
    (214, 218, "US", 36351),
    (219, 221, "KR", 4659),
    (222, 223, "CN", 4837),
]


class GeoIPResolver:
    """Offline GeoIP resolver for country and ASN lookup.

    Resolves IP addresses to country codes and ASN numbers using
    only local data. Never makes network calls.
    """

    def __init__(self, node_pool: list[NetworkNode] | None = None) -> None:
        """Initialize the resolver.

        Args:
            node_pool: Optional list of NetworkNode instances. If provided,
                their IP → country/ASN mappings take priority over the
                built-in range fallback.
        """
        self._ip_lookup: dict[str, tuple[str, int]] = {}
        if node_pool:
            for node in node_pool:
                self._ip_lookup[node.ip] = (node.country, node.asn)

    def lookup(self, ip: str) -> GeoIPResult:
        """Resolve an IP address to country and ASN.

        Args:
            ip: IPv4 address string (e.g. "45.33.32.156").

        Returns:
            A ``GeoIPResult`` with country, ASN, and resolution source.
        """
        # Strategy 1: Node pool lookup (exact match)
        if ip in self._ip_lookup:
            country, asn = self._ip_lookup[ip]
            return GeoIPResult(country=country, asn=asn, source="node_pool")

        # Strategy 2: Built-in range fallback
        try:
            first_octet = int(ip.split(".")[0])
        except (ValueError, IndexError):
            return GeoIPResult(country=None, asn=None, source="builtin_range")

        for start, end, country, asn in _BUILTIN_RANGES:
            if start <= first_octet <= end:
                return GeoIPResult(country=country, asn=asn, source="builtin_range")

        return GeoIPResult(country=None, asn=None, source="builtin_range")

    def lookup_country(self, ip: str) -> str | None:
        """Convenience: resolve IP to country code only."""
        return self.lookup(ip).country

    def lookup_asn(self, ip: str) -> int | None:
        """Convenience: resolve IP to ASN only."""
        return self.lookup(ip).asn

    @property
    def pool_size(self) -> int:
        """Number of IPs registered from the node pool."""
        return len(self._ip_lookup)
