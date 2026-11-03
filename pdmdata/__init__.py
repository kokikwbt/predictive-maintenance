"""Predictive-maintenance dataset catalog, downloads, and Polars loading."""

from pdmdata.catalog import format_summary, load_catalog, load_metadata, summary
from pdmdata.config import Settings, get_settings
from pdmdata.datasets import load, supported_verifications, verify
from pdmdata.io import (
    bulk_downloads,
    download,
    download_command,
    download_variants,
    supported_downloads,
)
from pdmdata.visualization import available_plots, visualize

__all__ = [
    "Settings",
    "get_settings",
    "download",
    "download_command",
    "download_variants",
    "bulk_downloads",
    "format_summary",
    "load_catalog",
    "load",
    "available_plots",
    "load_metadata",
    "summary",
    "supported_downloads",
    "supported_verifications",
    "verify",
    "visualize",
]
