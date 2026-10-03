"""Verify all local CARE v6 files and write an experiment inventory (no network)."""

from collections import Counter
from datetime import datetime, timezone
import json
from zipfile import ZipFile
import zlib

import polars as pl

from pdmdata.config import get_settings
from pdmdata.catalog import load_metadata
from .loader import load
from pdmdata.io.download import _validate_expected_files, _verify_checksum


def verify() -> dict:
    """Verify local files and all event loaders, then save an inventory and report."""
    metadata = load_metadata("care")
    directory = get_settings().data_root / "care"
    extracted = directory / "extracted"
    archive = directory / metadata["download"]["filename"]
    _verify_checksum(archive, metadata["download"]["checksum"])
    _validate_expected_files(extracted, metadata["download"]["expected_files"])

    # Verify the extracted bytes, not just the presence of filenames.
    with ZipFile(archive) as zipped:
        members = [member for member in zipped.infolist() if not member.is_dir()]
        for member in members:
            path = extracted / member.filename
            if path.stat().st_size != member.file_size:
                raise ValueError(f"Extracted size mismatch: {path}")
            crc = 0
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    crc = zlib.crc32(chunk, crc)
            if crc != member.CRC:
                raise ValueError(f"Extracted CRC mismatch: {path}")
        extracted_bytes = sum(member.file_size for member in members)

    inventory = []
    farm_summary = {}
    for farm, expected_count in (("A", 22), ("B", 15), ("C", 58)):
        farm_dir = extracted / "CARE_To_Compare" / f"Wind Farm {farm}"
        events = load(wind_farm=farm, table="events", lazy=False)
        features = load(wind_farm=farm, table="features", lazy=False)
        recordings = sorted((farm_dir / "datasets").glob("*.csv"), key=lambda p: int(p.stem))
        if len(recordings) != expected_count or set(events["event_id"].to_list()) != {int(p.stem) for p in recordings}:
            raise ValueError(f"Event table and recordings disagree for Wind Farm {farm}")
        farm_summary[farm] = {
            "recordings": len(recordings),
            "event_labels": dict(Counter(events["event_label"].to_list())),
            "feature_descriptions": features.height,
        }
        assets = set()
        for path in recordings:
            recording = f"Wind Farm {farm}/datasets/{path.name}"
            # Fully parse every column to catch dtype errors hidden by head()/len().
            frame = load(wind_farm=farm, event_id=int(path.stem), lazy=False)
            required = {"time_stamp", "asset_id", "id", "train_test", "status_type_id"}
            if not required.issubset(frame.columns):
                raise ValueError(f"Missing CARE columns: {path}")
            assets.update(frame["asset_id"].unique().to_list())
            inventory.append({
                "wind_farm": farm,
                "event_id": int(path.stem),
                "recording": recording,
                "rows": frame.height,
                "columns": frame.width,
                "bytes": path.stat().st_size,
                "train_rows": frame.filter(pl.col("train_test") == "train").height,
                "prediction_rows": frame.filter(pl.col("train_test") == "prediction").height,
            })
            del frame
            print(f"Verified {len(inventory)}/95: {recording}", flush=True)
        farm_summary[farm]["assets"] = len(assets)

    pl.DataFrame(inventory).write_csv(directory / "inventory.csv")
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source": metadata["source"]["doi_url"],
        "source_checksum": metadata["download"]["checksum"],
        "archive_bytes": archive.stat().st_size,
        "extracted_bytes": extracted_bytes,
        "verified_files": len(members),
        "recordings": len(inventory),
        "total_rows": sum(item["rows"] for item in inventory),
        "wind_farms": farm_summary,
        "checks": ["source checksum", "expected paths", "all extracted file sizes and CRCs", "event ID coverage", "all CSV columns parsed"],
    }
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
