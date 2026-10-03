# Real GeoIP, Tor and VPN data (offline setup)

ChainTrace never calls the network at runtime. To use real open IP data, download it **once on a connected machine**,
copy the files over, and point ChainTrace at them. Without them a coarse built-in first-octet table is used
(fine for the synthetic demo, not for real investigations).

## Files

Place in `backend/reference/geoip/` (or any directory named by `CHAINTRACE_GEOIP_DIR`):

| File | Source | Format |
|---|---|---|
| `dbip-country-lite.csv` | DB-IP "IP to Country Lite" (free, CC-BY 4.0, attribution required) - https://db-ip.com/db/download/ip-to-country-lite | `start_ip,end_ip,country` (no header) |
| `asn.csv` | DB-IP "IP to ASN Lite" (same licence), **or** MaxMind GeoLite2-ASN-Blocks-IPv4.csv (free account; GeoLite2 EULA) | `start_ip,end_ip,asn,org` or `network,asn,org` |

Decompress first (`gunzip dbip-country-lite-*.csv.gz`) and rename. IPv6 rows are ignored. A corrupt file is logged
and ignored; the built-in table takes over.

Confirm it loaded: the Forensics Lab status strip shows `GeoIP: dbip_country_lite+asn_csv`, and
`GET /forensics/status` reports `geoip_source`.

## Tor exit nodes

`backend/reference/tor_exit_nodes.txt`: one IPv4 per line, `#` comments allowed. The bundled file is a **synthetic sample**
(reserved documentation ranges). For real data, save the Tor Project bulk exit list (https://check.torproject.org/torbulkexitlist)
as this file.

## Hosting / VPN networks

`backend/reference/vpn_hosting_asns.txt`: one ASN per line. The bundled list holds well-known hosting providers
(M247, OVH, DigitalOcean, Hetzner, ...). Extend it with VPN providers you care about. ASN matching is a coarse signal:
plenty of legitimate traffic comes from hosting networks.

## Dataset columns

If your dataset already has `geo_country` and `asn` columns they are used as given; lookups only fill empty cells.
