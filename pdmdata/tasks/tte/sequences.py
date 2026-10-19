"""Model-independent sequence samples and versioned C-MAPSS handoffs.

Arrays use cycle units. Normalization is fitted on training entities only.
No torch, research-model imports, or automatic downloads are required.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CmapssSequenceSpec:
    """Reusable data settings; model and optimization settings are separate."""
    subset: str
    feature_columns: Sequence[str]
    seq_len: int = 30
    rul_cap: int | None = 125
    normalization: str = "train_minmax"
    validation_fraction: float = 0.2
    split_seed: int = 0
    train_stride: int = 5
    validation_stride: int = 5
    validation_units: Sequence[int] | None = None
    evaluation_units: Sequence[int] | None = None
    minimum_endpoint: int | None = None
    smoothing_alpha: float = 1.0
    short_test_policy: str = "repeat_first"

    def __post_init__(self):
        if self.subset not in ("FD001", "FD002", "FD003", "FD004"):
            raise ValueError("subset must be FD001, FD002, FD003, or FD004")
        for name in ("seq_len", "train_stride", "validation_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_endpoint is not None and (isinstance(self.minimum_endpoint, bool) or not isinstance(self.minimum_endpoint, int) or self.minimum_endpoint < 1):
            raise ValueError("minimum_endpoint must be a positive cycle or None")
        if self.rul_cap is not None and (
            isinstance(self.rul_cap, bool) or not isinstance(self.rul_cap, int) or self.rul_cap < 1
        ):
            raise ValueError("rul_cap must be a positive integer or None")
        if self.normalization not in ("train_minmax", "train_standard"):
            raise ValueError("normalization must be train_minmax or train_standard")
        if not np.isfinite(self.smoothing_alpha) or not 0 < self.smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.short_test_policy not in ("repeat_first", "linear_resample"):
            raise ValueError("unknown short test policy")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must satisfy 0 <= f < 1")
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be non-empty and unique")
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        for name in ("validation_units", "evaluation_units"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, tuple(values))

    def to_dict(self):
        values = asdict(self)
        values["feature_columns"] = list(self.feature_columns)
        for name in ("validation_units", "evaluation_units"):
            items = getattr(self, name)
            if items is None:
                values.pop(name)
            else:
                values[name] = list(items)
        if self.minimum_endpoint is None:
            values.pop("minimum_endpoint")
        if self.smoothing_alpha == 1.0:
            values.pop("smoothing_alpha")
        if self.short_test_policy == "repeat_first":
            values.pop("short_test_policy")
        return values


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_provenance():
    """Record the installed generator, including editable uncommitted sources."""
    import pdmdata
    root = Path(pdmdata.__file__).resolve().parents[1]
    commit = dirty = None
    # Wheels must not inherit the Git identity of an unrelated containing repo.
    if (root / ".git").exists():
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    sources = {str(p.relative_to(root)): _digest(p) for p in (root / "pdmdata").rglob("*.py")}
    return {"pdmdata_version": version("pdmdata"), "pdmdata_commit": commit,
            "pdmdata_dirty": dirty, "pdmdata_source_sha256": sources,
            "generator": "pdmdata.tasks.tte.sequences", "generator_sha256": _digest(__file__),
            "environment_sha256": {name: _digest(root / name) for name in ("pyproject.toml", "uv.lock")
                                   if (root / name).is_file()},
            "python": sys.version, "numpy": np.__version__, "polars": version("polars")}


def build_sequence_samples(frame, features, minimum, scale, length, stride, *,
                           targets=None, entity_column="unit_number", time_column="cycle",
                           target_column="time_to_event", minimum_endpoint=None,
                           smoothing_alpha=1.0, short_test_policy="repeat_first"):
    """Build past-only windows within integer-ID entities, retaining final rows.

    Explicit ``targets`` creates one scoring window per entity. Short series
    repeat their first value/time on the left. Other splits also retain the
    complete normalized point sequence; no future rows enter a window.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (length, stride)):
        raise ValueError("length and stride must be positive integers")
    features = list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be non-empty and unique")
    minimum, scale = np.asarray(minimum), np.asarray(scale)
    if minimum.shape != (len(features),) or scale.shape != minimum.shape:
        raise ValueError("normalization shape must match features")
    if not np.isfinite(minimum).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("normalization must be finite with positive scales")
    result = {k: [] for k in ("x", "y", "times", "entity", "endpoint")}
    points = {k: [] for k in ("x", "y", "entity")}
    if frame.is_empty():
        raise ValueError("sequence samples require at least one entity")
    for group in frame.sort([entity_column, time_column]).partition_by(entity_column, maintain_order=True):
        unit = int(group[entity_column][0])
        values = (group.select(features).to_numpy().astype(np.float32) - minimum) / scale
        if smoothing_alpha != 1.0:
            numerator = np.zeros(values.shape[1], dtype=np.float64)
            mass = 0.0
            smoothed = np.empty_like(values)
            for index, observed in enumerate(values):
                numerator = (1-smoothing_alpha)*numerator + smoothing_alpha*observed
                mass = (1-smoothing_alpha)*mass + smoothing_alpha
                smoothed[index] = numerator / mass
            values = smoothed
        clock = group[time_column].to_numpy().astype(np.float32)
        if not np.isfinite(values).all() or not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
            raise ValueError("observations must be finite and time strictly increasing per entity")
        if targets is None:
            labels = group[target_column].to_numpy().astype(np.float32)
            if not np.isfinite(labels).all() or (labels < 0).any():
                raise ValueError("remaining-time labels must be finite and non-negative")
            points["x"].extend(values)
            points["y"].extend(labels)
            points["entity"].extend([unit] * len(values))
            ends = list(range(length, len(values) + 1, stride))
            if not ends or ends[-1] != len(values):
                ends.append(len(values))
        else:
            target = targets[unit]
            if not np.isfinite(target) or target < 0:
                raise ValueError("scoring targets must be finite and non-negative")
            ends = [len(values)]
        for end in ends:
            if minimum_endpoint is not None and clock[end-1] < minimum_endpoint:
                continue
            begin = max(0, end - length)
            x, times = values[begin:end], clock[begin:end]
            if len(x) < length:
                if targets is not None and short_test_policy == "linear_resample" and len(x) > 1:
                    old = np.arange(len(x), dtype=np.float64)
                    coefficients = np.polyfit(old, x, deg=1)
                    new = np.arange(length, dtype=np.float64)*len(x)/length
                    x = (new[:, None]*coefficients[0]+coefficients[1]).astype(np.float32)
                    times = np.linspace(clock[begin], clock[end-1], length, dtype=np.float32)
                else:
                    pad = length - len(x)
                    x = np.concatenate((np.repeat(x[:1], pad, 0), x))
                    times = np.concatenate((np.repeat(times[:1], pad), times))
            result["x"].append(x)
            result["times"].append(times)
            result["y"].append(targets[unit] if targets is not None else labels[end - 1])
            result["entity"].append(unit)
            result["endpoint"].append(clock[end - 1])
    if not result["x"]:
        raise ValueError("no samples meet the endpoint policy")
    arrays = {k: np.asarray(v, dtype=np.int64 if k == "entity" else np.float32) for k, v in result.items()}
    arrays["mask"] = np.ones_like(arrays["x"])
    if targets is None:
        arrays.update({"points_" + k: np.asarray(v, dtype=np.int64 if k == "entity" else np.float32)
                       for k, v in points.items()})
    return arrays


def prepare_cmapss_sequences(spec: CmapssSequenceSpec | Mapping, *, include_test=True):
    """Prepare saved data with train-only scaling and optional evaluation IDs.

    ``evaluation_units`` reserves engines from the official training cohort
    before the inner validation split. ``include_test=False`` never reads the
    official test series or its labels. Evaluation labels remain scoring-only.
    """
    from pdmdata.datasets.cmapss import (
        load as load_cmapss,
        rul as cmapss_test_end_rul,
    )
    from pdmdata.tasks.cmapss_common import (
        feature_columns,
        label_failed_rows,
    )
    from pdmdata.tasks.splits import filter_entities, split_entities
    if not isinstance(spec, CmapssSequenceSpec):
        spec = CmapssSequenceSpec(**spec)
    raw = load_cmapss(spec.subset, "train", with_rul=True)
    labeled = label_failed_rows(raw, time_cap=spec.rul_cap)
    candidates = feature_columns(labeled, include_operating_settings=True, drop_constant_features=False)
    unknown = set(spec.feature_columns) - set(candidates)
    if unknown:
        raise ValueError(f"feature_columns must contain only candidate sensors/settings: {sorted(unknown)}")
    units = set(labeled["unit_number"].unique().to_list())
    evaluation = set(spec.evaluation_units or ())
    if evaluation - units or evaluation == units:
        raise ValueError("evaluation_units must reserve a proper subset of training-cohort engines")
    if evaluation & set(spec.validation_units or ()):
        raise ValueError("validation and evaluation entities overlap")
    train_units, validation = split_entities(sorted(units-evaluation),
        validation_fraction=spec.validation_fraction, random_state=spec.split_seed,
        validation_entities=spec.validation_units)
    train_frame = filter_entities(labeled, train_units, "unit_number")
    train = train_frame.select(list(spec.feature_columns)).to_numpy().astype(np.float32)
    if spec.normalization == "train_standard":
        minimum, scale = train.mean(0), train.std(0)
    else:
        minimum, scale = train.min(0), np.ptp(train, axis=0)
    scale[scale == 0] = 1
    frames = [("train", train_frame)]
    if validation:
        frames.append(("validation", filter_entities(labeled, validation, "unit_number")))
    if evaluation:
        frames.append(("evaluation", filter_entities(labeled, sorted(evaluation), "unit_number")))
    targets = None
    if include_test:
        test = load_cmapss(spec.subset, "test", with_rul=False)
        end = cmapss_test_end_rul(spec.subset)
        labels = end["RUL"].to_numpy()
        if spec.rul_cap is not None:
            labels = np.minimum(labels, spec.rul_cap)
        targets = dict(zip(end["unit_number"].to_list(), labels.tolist()))
        frames.append(("test", test))
    splits = {}
    for name, frame in frames:
        splits[name] = build_sequence_samples(frame, spec.feature_columns, minimum, scale, spec.seq_len,
            spec.train_stride if name == "train" else spec.validation_stride,
            targets=targets if name == "test" else None, minimum_endpoint=spec.minimum_endpoint if name != "test" else None,
            smoothing_alpha=spec.smoothing_alpha, short_test_policy=spec.short_test_policy)
    metadata = {"schema_version": 1, "dataset": spec.subset, "data_config": spec.to_dict(),
        "time_unit": "cycles", "include_test": bool(include_test),
        "normalization": {"fitted_on": "train_only", "minimum": minimum.tolist(), "scale": scale.tolist()},
        "split_entities": {name: sorted(set(arrays["entity"].tolist())) for name, arrays in splits.items()},
        "n_samples": {name: len(arrays["y"]) for name, arrays in splits.items()}}
    return splits, metadata


def export_cmapss_sequences(output, spec: CmapssSequenceSpec | Mapping, *, include_test=True):
    """Create a new NPZ handoff directory with source and output hashes."""
    import pdmdata
    splits, manifest = prepare_cmapss_sequences(spec, include_test=include_test)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, arrays in splits.items():
        np.savez_compressed(output / f"{name}.npz", **arrays)
    subset = manifest["dataset"]
    raw_root = pdmdata.get_settings().data_root / "cmapss"
    expected = {f"train_{subset}.txt"}
    if include_test:
        expected.update((f"test_{subset}.txt", f"RUL_{subset}.txt"))
    files = [p for p in raw_root.rglob("*.txt") if p.name in expected]
    if len(files) != len(expected) or {p.name for p in files} != expected:
        raise ValueError("cannot uniquely identify source C-MAPSS files")
    manifest.update(sequence_provenance())
    manifest.update({"files": {f"{name}.npz": _digest(output / f"{name}.npz") for name in splits},
                     "source_data_sha256": {p.name: _digest(p) for p in files}, "command": sys.argv})
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def complete_sequences(arrays, cap):
    """Verify a legacy full-point handoff of complete cycles 1..n per entity.

    This derivation is for completed, contiguous C-MAPSS training cohorts.
    Truncated test, irregular clocks, and reordered points are rejected.
    """
    sequences = {}
    for unit in np.unique(arrays["entity"]):
        x = arrays["points_x"][arrays["points_entity"] == unit]
        labels = arrays["points_y"][arrays["points_entity"] == unit]
        raw = len(x) - np.arange(1, len(x) + 1)
        expected = raw if cap is None else np.minimum(raw, cap)
        if not np.array_equal(labels, expected):
            raise ValueError("complete sequence labels or point order do not match")
        for i in np.flatnonzero(arrays["entity"] == unit):
            end = int(arrays["endpoint"][i])
            length = arrays["x"].shape[1]
            if end != arrays["endpoint"][i] or end < length or end > len(x):
                raise ValueError("handoff does not contain complete unpadded windows")
            if not np.array_equal(arrays["times"][i], np.arange(end-length+1, end+1)):
                raise ValueError("cycles are missing or out of order")
            target = len(x)-end if cap is None else min(len(x)-end, cap)
            if not np.array_equal(arrays["x"][i], x[end-length:end]) or arrays["y"][i] != target:
                raise ValueError("point sequence and existing window do not match")
        sequences[int(unit)] = x
    return sequences


def common_windows(arrays, sequences, length, minimum_endpoint):
    """Keep existing endpoint order and build past-only complete windows."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (length, minimum_endpoint)) or minimum_endpoint < length:
        raise ValueError("minimum_endpoint must be at least the positive window length")
    indices = np.flatnonzero(arrays["endpoint"] >= minimum_endpoint)
    if not len(indices):
        raise ValueError("no shared endpoints meet the minimum cycle")
    x = np.stack([sequences[int(arrays["entity"][i])][int(arrays["endpoint"][i])-length:int(arrays["endpoint"][i])]
                  for i in indices])
    return x.astype(np.float64), arrays["y"][indices].astype(np.float64), arrays["entity"][indices], arrays["endpoint"][indices]


def export_derived_windows(source, output, *, length, minimum_endpoint):
    """Derive train/validation windows from a verified NPZ handoff.

    Preserve its split, normalization, labels, points, and endpoint order.
    Never open or copy test.npz. Record the parent and the new generator
    separately so the original raw-data processing provenance is retained.
    """
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["time_unit"] != "cycles":
        raise ValueError("expected schema 1 with cycle-unit times")
    if set(manifest["split_entities"]["train"]) & set(manifest["split_entities"]["validation"]):
        raise ValueError("training and validation entities overlap")
    arrays = {}
    for name in ("train", "validation"):
        path = source / f"{name}.npz"
        if _digest(path) != manifest["files"][path.name]:
            raise ValueError("source handoff hash does not match")
        with np.load(path, allow_pickle=False) as data:
            a = {k: data[k].copy() for k in data.files}
        if sorted(set(a["entity"].tolist())) != manifest["split_entities"][name]:
            raise ValueError("source entity split does not match")
        if not all(np.isfinite(v).all() for v in a.values()) or not (a["mask"] == 1).all():
            raise ValueError("derivation requires finite complete observations")
        sequences = complete_sequences(a, manifest["data_config"]["rul_cap"])
        x, y, u, t = common_windows(a, sequences, length, minimum_endpoint)
        clock = t[:, None] - np.arange(length-1, -1, -1)[None, :]
        points = {k: v for k, v in a.items() if k.startswith("points_")}
        arrays[name] = dict(x=x.astype(np.float32), y=y.astype(np.float32), entity=u, endpoint=t,
                            times=clock.astype(np.float32), mask=np.ones_like(x, dtype=np.float32), **points)
    output.mkdir(parents=True, exist_ok=False)
    for name, a in arrays.items():
        np.savez_compressed(output / f"{name}.npz", **a)
    derived = {**manifest, "data_config": {**manifest["data_config"], "seq_len": length,
                                         "common_endpoint_minimum": minimum_endpoint},
        "files": {f"{name}.npz": _digest(output / f"{name}.npz") for name in arrays},
        "n_samples": {name: len(a["y"]) for name, a in arrays.items()},
        "derived_from_manifest_sha256": _digest(source / "manifest.json"),
        "derivation": "verified complete point handoff; past-only windows; same endpoints; no test split generated",
        "sample_generator": sequence_provenance()}
    (output / "manifest.json").write_text(
        json.dumps(derived, indent=2) + "\n", encoding="utf-8"
    )
    return derived
