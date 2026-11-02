"""Shared synthetic fixtures for dataset loader and task tests."""

from tests.fixtures.ncmapss import write_rich_fixture, write_tiny_fixture
from tests.fixtures.xjtu_sy import write_bearing

__all__ = [
    "write_bearing",
    "write_rich_fixture",
    "write_tiny_fixture",
]
