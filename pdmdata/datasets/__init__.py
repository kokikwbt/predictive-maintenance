"""Dataset adapters and common load() / verify() dispatch entry points."""

from pdmdata.datasets.load import load, supported_verifications, verify

__all__ = ["load", "supported_verifications", "verify"]
