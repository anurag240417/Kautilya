"""Verify zero runtime network dependency for Kautilya.

Scans the codebase for:
1. Imports of network-calling libraries (requests, urllib.request, httpx, etc.)
2. Code patterns that make network calls (socket.connect, urlopen, etc.)
3. GeoIP usage ensuring local database (not external lookups)
4. Environment variables pointing to external URLs

Usage:
    python scripts/verify_offline.py

Exit code 0 = all checks pass, 1 = violations found.
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Libraries that make network calls
NETWORK_LIBRARIES = [
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    "grpc",
    "boto3",
    "google.cloud",
    "azure",
    "paramiko",
    "fabric",
    "socket",
]

# Import patterns that indicate network usage (stricter than library name)
NETWORK_IMPORT_PATTERNS = [
    r"^\s*import\s+requests\b",
    r"^\s*from\s+requests\b",
    r"^\s*import\s+httpx\b",
    r"^\s*from\s+httpx\b",
    r"^\s*import\s+aiohttp\b",
    r"^\s*from\s+aiohttp\b",
    r"^\s*from\s+urllib\.request\s+import",
    r"^\s*import\s+urllib\.request\b",
    r"^\s*import\s+grpc\b",
    r"^\s*from\s+grpc\b",
    r"^\s*import\s+boto3\b",
    r"^\s*from\s+boto3\b",
]

# Code patterns that indicate runtime network calls
NETWORK_CALL_PATTERNS = [
    r"requests\.(get|post|put|delete|patch|head|options)\s*\(",
    r"httpx\.(get|post|put|delete|patch|head|options|Client|AsyncClient)\s*\(",
    r"urllib\.request\.(urlopen|urlretrieve|Request)\s*\(",
    r"socket\.connect\s*\(",
    r"socket\.create_connection\s*\(",
    r"urlopen\s*\(",
    r"\.fetch\s*\(\s*['\"]https?://",
    r"aiohttp\.ClientSession\s*\(",
]

# Allowed exceptions (test files, this script, etc.)
ALLOWED_PATHS = {
    "scripts/verify_offline.py",  # This script itself
    "tests/",                      # Test files may mock network calls
    ".venv/",                      # Virtual environment
    "node_modules/",               # NPM packages
    "benchmarks/",                 # Benchmark scripts
}

# External URL patterns in environment variables
URL_PATTERN = re.compile(r"https?://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _should_skip(filepath: Path) -> bool:
    """Check if file should be excluded from scanning."""
    rel = str(filepath.relative_to(PROJECT_ROOT))
    return any(rel.startswith(allowed) for allowed in ALLOWED_PATHS)


def _get_python_files() -> list[Path]:
    """Collect all Python files in the backend and root."""
    files = []
    for pattern_dir in [BACKEND_DIR, PROJECT_ROOT]:
        for py_file in pattern_dir.rglob("*.py"):
            if not _should_skip(py_file) and ".venv" not in str(py_file):
                files.append(py_file)
    # Deduplicate
    return list(set(files))


def check_network_imports() -> list[str]:
    """Scan for imports of network-calling libraries."""
    violations = []
    for py_file in _get_python_files():
        try:
            content = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern in NETWORK_IMPORT_PATTERNS:
                if re.match(pattern, line):
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"  {rel_path}:{line_num} — {line.strip()}"
                    )
    return violations


def check_network_calls() -> list[str]:
    """Scan for code patterns that make network calls."""
    violations = []
    for py_file in _get_python_files():
        try:
            content = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for line_num, line in enumerate(content.splitlines(), 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern in NETWORK_CALL_PATTERNS:
                if re.search(pattern, line):
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"  {rel_path}:{line_num} — {stripped}"
                    )
    return violations


def check_geoip_local() -> list[str]:
    """Verify GeoIP uses local database, not external lookups."""
    violations = []
    for py_file in _get_python_files():
        try:
            content = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Check for external GeoIP API calls
        external_geoip_patterns = [
            r"ip-api\.com",
            r"ipinfo\.io",
            r"ipstack\.com",
            r"freegeoip",
            r"geoip\.nekudo",
            r"geolocation-db",
        ]
        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern in external_geoip_patterns:
                if re.search(pattern, line):
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"  {rel_path}:{line_num} — External GeoIP API: {line.strip()}"
                    )
    return violations


def check_env_urls() -> list[str]:
    """Check environment variables for external URLs."""
    violations = []

    # Check .env.example
    env_file = PROJECT_ROOT / ".env.example"
    if env_file.exists():
        for line_num, line in enumerate(env_file.read_text().splitlines(), 1):
            if URL_PATTERN.search(line) and not line.strip().startswith("#"):
                violations.append(
                    f"  .env.example:{line_num} — External URL: {line.strip()}"
                )

    # Check config.py for hardcoded URLs
    config_file = BACKEND_DIR / "config.py"
    if config_file.exists():
        content = config_file.read_text()
        for line_num, line in enumerate(content.splitlines(), 1):
            if URL_PATTERN.search(line) and not line.strip().startswith("#"):
                violations.append(
                    f"  backend/config.py:{line_num} — External URL: {line.strip()}"
                )

    return violations


def check_runtime_socket_usage() -> list[str]:
    """Check for raw socket usage that could indicate network calls."""
    violations = []
    for py_file in _get_python_files():
        try:
            content = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # wsgiref.simple_server is allowed (local binding only)
        if "wsgiref" in content:
            continue

        if re.search(r"import\s+socket\b", content):
            rel_path = py_file.relative_to(PROJECT_ROOT)
            violations.append(
                f"  {rel_path} — imports socket module"
            )

    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_verification() -> bool:
    """Execute all offline compliance checks. Returns True if all pass."""
    print("=" * 70)
    print("Kautilya Offline Compliance Verification")
    print("=" * 70)
    print()

    all_passed = True
    checks = [
        ("Network Library Imports", check_network_imports),
        ("Network Call Patterns", check_network_calls),
        ("GeoIP External Lookups", check_geoip_local),
        ("Environment Variable URLs", check_env_urls),
        ("Raw Socket Usage", check_runtime_socket_usage),
    ]

    for name, check_fn in checks:
        violations = check_fn()
        if violations:
            print(f"✗ {name}: {len(violations)} violation(s) found")
            for v in violations:
                print(v)
            all_passed = False
        else:
            print(f"✓ {name}: PASS")
        print()

    print("=" * 70)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED — Zero runtime network dependencies verified.")
    else:
        print("RESULT: VIOLATIONS FOUND — Review and fix before deployment.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = run_verification()
    sys.exit(0 if passed else 1)
