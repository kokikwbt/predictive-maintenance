"""Predictive-maintenance dataset catalog, downloads, and Polars loading."""

from .catalog import format_summary, load_catalog, load_metadata, summary
from .download import (
    bulk_downloads,
    download,
    download_command,
    download_variants,
    supported_downloads,
)
from .loaders import load
from .visualization import available_plots, visualize

__all__ = [
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
    "visualize",
]
