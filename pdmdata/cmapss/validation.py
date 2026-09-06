"""Complete C-MAPSS archive inventory checks."""

from .loader import inventory

# Counts measured from the NASA-distributed files, including the FD004 correction.
EXPECTED = {
    ("FD001", "train"): (100, 20631), ("FD001", "test"): (100, 13096),
    ("FD002", "train"): (260, 53759), ("FD002", "test"): (259, 33991),
    ("FD003", "train"): (100, 24720), ("FD003", "test"): (100, 16596),
    ("FD004", "train"): (249, 61249), ("FD004", "test"): (248, 41214),
}


def verify():
    """Validate all 12 data files and compare complete row/engine counts."""
    report = inventory()
    for row in report.iter_rows(named=True):
        key = (row["subset"], row["split"])
        if (row["units"], row["rows"]) != EXPECTED[key]:
            raise ValueError(f"Incomplete or changed C-MAPSS {key}: expected {EXPECTED[key]} units/rows")
    return report
