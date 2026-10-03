"""XJTU-SY bearing run-to-failure trajectories and derived RUL labels."""

from .loader import (
    available_bearings,
    inventory,
    load,
    load_many,
)

__all__ = ["available_bearings", "inventory", "load", "load_many"]
