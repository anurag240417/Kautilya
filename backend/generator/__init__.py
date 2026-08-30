"""Synthetic network data generator module.

Generates synthetic network observations (IP, port, ASN, country,
propagation timestamps) for Elliptic++ transactions. All generated
data carries ``is_synthetic=True``.

Must use reproducible random seeds and record generation configuration.
"""
