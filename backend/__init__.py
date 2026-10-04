"""Kautilya backend package."""

import os as _os

# Backward compatibility: the project used to be called ChainTrace and read CHAINTRACE_* settings.
# Existing deployments (for example environment variables already set on a hosting dashboard) keep
# working: a legacy variable is copied to its KAUTILYA_ name unless the new name is already set.
for _key, _value in list(_os.environ.items()):
    if _key.startswith("CHAINTRACE_"):
        _os.environ.setdefault("KAUTILYA_" + _key.removeprefix("CHAINTRACE_"), _value)
