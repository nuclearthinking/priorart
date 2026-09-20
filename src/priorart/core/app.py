"""Application identity: the version every handshake and profile pins.

One definition point so the MCP handshake, store profiles, the daemon
protocol and doctor all agree without depending on a storage internal.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    APP_VERSION = version("priorart")
except PackageNotFoundError:  # running from a source checkout without installation
    APP_VERSION = "unknown"
