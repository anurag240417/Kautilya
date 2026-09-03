"""Entity resolution and candidate linkage.

Resolves relationships between wallets, transactions, and IPs to identify
potential investigative links without asserting definitive real-world attribution.

Forensic Rule:
    Correlation does NOT imply attribution. Temporal proximity and network
    observations alone must NOT be treated as proof of wallet or IP ownership.
    Wallet-to-IP relationships represent candidate investigative associations
    and retain `is_synthetic=True`.
"""

from collections import defaultdict

from pydantic import BaseModel, Field

from backend.domain.correlation import CandidateIPRole, TransactionIPCorrelation


class WalletIPCorrelation(BaseModel):
    """Candidate investigative association between a wallet address and an IP.

    Derived from transactions involving the wallet address that correlate
    with network propagation observations.
    """

    address: str = Field(description="Bitcoin wallet address (Elliptic++)")
    ip: str = Field(description="Candidate IP address (synthetic)")
    associated_txids: list[int] = Field(
        description="Transactions linking this wallet to the IP"
    )
    candidate_roles: list[CandidateIPRole] = Field(
        description="Observed roles of this IP for the associated transactions"
    )
    correlation_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Aggregated confidence for the wallet-IP relationship [0.0, 1.0]",
    )
    is_synthetic: bool = Field(
        default=True,
        description="Always True. Derived from synthetic network correlation.",
    )
    disclaimer: str = Field(
        default="Temporal correlation alone must NOT be treated as proof of wallet/IP ownership.",
        description="Forensic evidentiary disclaimer",
    )


def resolve_wallet_to_candidate_ips(
    address: str,
    tx_ip_correlations: list[TransactionIPCorrelation],
    wallet_tx_map: dict[str, list[int]],
) -> list[WalletIPCorrelation]:
    """Resolve candidate IPs associated with a given wallet address.

    Aggregates transaction-level correlations across all transactions involving
    the wallet address.

    Args:
        address: Wallet address to investigate.
        tx_ip_correlations: List of computed TransactionIPCorrelation records.
        wallet_tx_map: Mapping of wallet address → list of associated txids.

    Returns:
        List of WalletIPCorrelation records for the address.
    """
    txids = set(wallet_tx_map.get(address, []))
    if not txids:
        return []

    # Group correlations by IP for the matching txids
    ip_groups: dict[str, list[TransactionIPCorrelation]] = defaultdict(list)
    for corr in tx_ip_correlations:
        if corr.txid in txids:
            ip_groups[corr.ip].append(corr)

    results: list[WalletIPCorrelation] = []
    for ip, corrs in ip_groups.items():
        associated_txids = sorted(list({c.txid for c in corrs}))
        roles = sorted(list({c.role for c in corrs}))
        # Aggregate confidence: max or mean confidence
        mean_conf = sum(c.correlation_confidence for c in corrs) / len(corrs)

        results.append(
            WalletIPCorrelation(
                address=address,
                ip=ip,
                associated_txids=associated_txids,
                candidate_roles=roles,
                correlation_confidence=mean_conf,
                is_synthetic=True,
            )
        )

    results.sort(key=lambda w: -w.correlation_confidence)
    return results


def resolve_ip_to_candidate_wallets(
    ip: str,
    tx_ip_correlations: list[TransactionIPCorrelation],
    tx_wallet_map: dict[int, list[str]],
) -> list[WalletIPCorrelation]:
    """Resolve candidate wallets associated with a given IP address.

    Args:
        ip: Target IP address.
        tx_ip_correlations: Computed TransactionIPCorrelation records.
        tx_wallet_map: Mapping of txid → list of participating wallet addresses.

    Returns:
        List of WalletIPCorrelation records for the IP.
    """
    ip_corrs = [c for c in tx_ip_correlations if c.ip == ip]
    if not ip_corrs:
        return []

    wallet_groups: dict[str, list[TransactionIPCorrelation]] = defaultdict(list)
    for corr in ip_corrs:
        wallets = tx_wallet_map.get(corr.txid, [])
        for w in wallets:
            wallet_groups[w].append(corr)

    results: list[WalletIPCorrelation] = []
    for address, corrs in wallet_groups.items():
        associated_txids = sorted(list({c.txid for c in corrs}))
        roles = sorted(list({c.role for c in corrs}))
        mean_conf = sum(c.correlation_confidence for c in corrs) / len(corrs)

        results.append(
            WalletIPCorrelation(
                address=address,
                ip=ip,
                associated_txids=associated_txids,
                candidate_roles=roles,
                correlation_confidence=mean_conf,
                is_synthetic=True,
            )
        )

    results.sort(key=lambda w: -w.correlation_confidence)
    return results
