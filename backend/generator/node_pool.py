"""Network node/IP pool generation.

Creates a pool of synthetic network nodes with assigned IP addresses,
ASN numbers, and geographic locations.

All nodes are synthetic — country and ASN are assigned directly from
weighted distributions rather than looked up via GeoIP, since every
record is fabricated.

See DATASET.md §9 and CONTEXT.md §3 for generation rules.
"""

import random
from enum import StrEnum

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    """Classification of synthetic Bitcoin P2P network nodes."""

    FULL_NODE = "full_node"
    RELAY = "relay"
    TOR_EXIT = "tor_exit"
    VPN = "vpn"


class NetworkNode(BaseModel):
    """A synthetic Bitcoin P2P network node.

    Represents a single node in the simulated network with its
    network identity and geographic metadata.
    """

    node_id: int = Field(description="Unique node identifier within the pool")
    ip: str = Field(description="Synthetic IPv4 address")
    port: int = Field(default=8333, ge=0, le=65535, description="Bitcoin P2P port")
    asn: int = Field(description="Autonomous System Number")
    country: str = Field(description="ISO 3166-1 alpha-2 country code")
    node_type: NodeType = Field(default=NodeType.FULL_NODE)


# --- Realistic distributions ---

# Top Bitcoin node countries with approximate real-world weights.
# Source: rough approximation of Bitnodes country distribution.
_COUNTRY_WEIGHTS: list[tuple[str, float]] = [
    ("US", 0.25),
    ("DE", 0.12),
    ("FR", 0.08),
    ("NL", 0.06),
    ("GB", 0.05),
    ("CA", 0.04),
    ("SG", 0.03),
    ("JP", 0.03),
    ("AU", 0.03),
    ("CH", 0.02),
    ("RU", 0.02),
    ("SE", 0.02),
    ("FI", 0.02),
    ("HK", 0.02),
    ("KR", 0.02),
    ("BR", 0.02),
    ("IN", 0.02),
    ("IT", 0.02),
    ("PL", 0.02),
    ("RO", 0.01),
    ("CZ", 0.01),
    ("AT", 0.01),
    ("IE", 0.01),
    ("ES", 0.01),
    ("UA", 0.01),
    # Remaining weight is spread across other countries
]

# Major ASNs associated with each country (simplified mapping).
# Each country maps to a list of (ASN, weight) tuples.
_COUNTRY_ASNS: dict[str, list[tuple[int, float]]] = {
    "US": [(15169, 0.3), (14618, 0.2), (20940, 0.15), (7922, 0.1), (16509, 0.25)],
    "DE": [(24940, 0.4), (51167, 0.2), (3320, 0.2), (6724, 0.2)],
    "FR": [(16276, 0.4), (12876, 0.3), (5410, 0.3)],
    "NL": [(60781, 0.3), (20857, 0.3), (1101, 0.2), (49981, 0.2)],
    "GB": [(5089, 0.3), (20712, 0.3), (2856, 0.2), (6453, 0.2)],
    "CA": [(16509, 0.3), (577, 0.3), (812, 0.2), (6327, 0.2)],
    "SG": [(16509, 0.4), (4657, 0.3), (38001, 0.3)],
    "JP": [(2497, 0.3), (17676, 0.3), (4713, 0.2), (9370, 0.2)],
    "AU": [(4764, 0.3), (1221, 0.3), (7545, 0.2), (4826, 0.2)],
    "CH": [(13030, 0.4), (6730, 0.3), (15600, 0.3)],
    "RU": [(49505, 0.3), (12389, 0.3), (31133, 0.2), (8492, 0.2)],
    "SE": [(29518, 0.4), (3301, 0.3), (8473, 0.3)],
    "FI": [(24940, 0.4), (1759, 0.3), (16086, 0.3)],
    "HK": [(4515, 0.4), (9304, 0.3), (4760, 0.3)],
    "KR": [(4766, 0.3), (9318, 0.3), (3786, 0.2), (4659, 0.2)],
    "BR": [(28573, 0.3), (8167, 0.3), (7738, 0.2), (16735, 0.2)],
    "IN": [(9498, 0.3), (55836, 0.3), (18209, 0.2), (45609, 0.2)],
    "IT": [(12874, 0.3), (30722, 0.3), (3269, 0.2), (12637, 0.2)],
    "PL": [(5617, 0.3), (12741, 0.3), (21021, 0.2), (29535, 0.2)],
    "RO": [(8708, 0.4), (6718, 0.3), (9050, 0.3)],
    "CZ": [(25248, 0.4), (5588, 0.3), (29208, 0.3)],
    "AT": [(8447, 0.4), (1764, 0.3), (6830, 0.3)],
    "IE": [(15502, 0.4), (5466, 0.3), (60233, 0.3)],
    "ES": [(12715, 0.4), (3352, 0.3), (12479, 0.3)],
    "UA": [(13188, 0.4), (15895, 0.3), (35213, 0.3)],
}

# Default ASN list for countries not in the mapping above.
_DEFAULT_ASNS: list[tuple[int, float]] = [
    (64496, 0.3), (64497, 0.3), (64498, 0.2), (64499, 0.2),
]

# Node type distribution weights.
_NODE_TYPE_WEIGHTS: list[tuple[NodeType, float]] = [
    (NodeType.FULL_NODE, 0.72),
    (NodeType.RELAY, 0.20),
    (NodeType.TOR_EXIT, 0.05),
    (NodeType.VPN, 0.03),
]

# IPv4 first-octet ranges to avoid (reserved/private/special-use).
_RESERVED_FIRST_OCTETS: set[int] = {
    0, 10, 100, 127, 169, 172, 192, 198, 203, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
    235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
    245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
}


def _generate_ip(rng: random.Random, used_ips: set[str]) -> str:
    """Generate a unique synthetic public IPv4 address.

    Avoids reserved/private ranges and ensures uniqueness
    within the current generation run.
    """
    while True:
        first = rng.randint(1, 223)
        if first in _RESERVED_FIRST_OCTETS:
            continue
        ip = f"{first}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        if ip not in used_ips:
            used_ips.add(ip)
            return ip


def _pick_weighted(rng: random.Random, items: list[tuple]) -> object:
    """Pick an item from a weighted list using the provided RNG."""
    values = [item[0] for item in items]
    weights = [item[1] for item in items]
    return rng.choices(values, weights=weights, k=1)[0]


def _pick_country(rng: random.Random) -> str:
    """Pick a country code from the weighted distribution.

    Countries not in the explicit list get a small residual probability.
    """
    return _pick_weighted(rng, _COUNTRY_WEIGHTS)


def _pick_asn(rng: random.Random, country: str) -> int:
    """Pick an ASN appropriate for the given country."""
    asn_list = _COUNTRY_ASNS.get(country, _DEFAULT_ASNS)
    return _pick_weighted(rng, asn_list)


def _pick_node_type(rng: random.Random) -> NodeType:
    """Pick a node type from the weighted distribution."""
    return _pick_weighted(rng, _NODE_TYPE_WEIGHTS)


def generate_node_pool(size: int, seed: int) -> list[NetworkNode]:
    """Generate a pool of synthetic Bitcoin P2P network nodes.

    Each node gets a unique IP, an ASN and country drawn from
    realistic weighted distributions, and a node type.

    Args:
        size: Number of nodes to generate. Must be positive.
        seed: Random seed for reproducible generation.

    Returns:
        List of ``NetworkNode`` instances.

    Raises:
        ValueError: If size is not positive.
    """
    if size <= 0:
        msg = f"Pool size must be positive, got {size}"
        raise ValueError(msg)

    rng = random.Random(seed)
    used_ips: set[str] = set()
    nodes: list[NetworkNode] = []

    for node_id in range(size):
        country = _pick_country(rng)
        node = NetworkNode(
            node_id=node_id,
            ip=_generate_ip(rng, used_ips),
            asn=_pick_asn(rng, country),
            country=country,
            node_type=_pick_node_type(rng),
        )
        nodes.append(node)

    return nodes
