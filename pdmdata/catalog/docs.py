"""Generate human-readable documentation from dataset metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from pdmdata.catalog import DATASETS_ROOT, ROOT, load_catalog
from pdmdata.tasks import TASKS, TASK_BY_ID


BEGIN = "<!-- BEGIN GENERATED METADATA -->"
END = "<!-- END GENERATED METADATA -->"
CATALOG_BEGIN = "<!-- BEGIN GENERATED DATASET TABLE -->"
CATALOG_END = "<!-- END GENERATED DATASET TABLE -->"
TASKS_BEGIN = "<!-- BEGIN GENERATED TASK TABLE -->"
TASKS_END = "<!-- END GENERATED TASK TABLE -->"


def _yes_no(value: Any) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if value else "No"


def _link(label: str, url: str) -> str:
    return "[{}]({})".format(label, url) if url else label


def render_metadata(metadata: Dict[str, Any]) -> str:
    """Render the generated section of a dataset README."""
    source = metadata["source"]
    license_info = metadata["license"]
    availability = metadata["availability"]
    characteristics = metadata.get("characteristics", {})

    lines = [
        BEGIN,
        "",
        "## Dataset overview",
        "",
        metadata["summary"],
        "",
        "| Item | Details |",
        "|---|---|",
        "| ID | `{}` |".format(metadata["id"]),
        "| Name | {} |".format(metadata["name"]),
        "| Provider | {} |".format(
            _link(source["name"], source.get("landing_page", ""))
        ),
        "| DOI | {} |".format(
            _link(source.get("doi") or "—", source.get("doi_url", ""))
        ),
        "| Availability | {} (checked: {}) |".format(
            availability["status"], availability["checked_at"]
        ),
        "| Access | {} |".format(availability["access"]),
        "| License | {} |".format(
            _link(license_info["name"], license_info.get("url", ""))
        ),
        "| Commercial use | {} |".format(_yes_no(license_info.get("commercial_use"))),
        "| Redistribution | {} |".format(_yes_no(license_info.get("redistribution"))),
        "| Data type | {} |".format(characteristics.get("kind", "—")),
        "| Feature dimensions | {} |".format(
            characteristics.get("feature_dimensions", {}).get("summary", "—")
        ),
        "| Feature counting | {} |".format(
            characteristics.get("feature_dimensions", {}).get("counting", "—")
        ),
        "| Feature-count source | {} |".format(
            ", ".join(
                _link(item["name"], item["url"])
                for item in characteristics.get("feature_dimensions", {}).get("sources", [])
            ) or "—"
        ),
        "| Tasks | {} |".format(", ".join(metadata.get("tasks", []))),
        "",
        "## Experiment-task suitability",
        "",
        "| Task | Support |",
        "|---|---|",
    ]
    support_labels = {"direct": "Direct", "derived": "Requires target derivation"}
    for task_id, level in metadata.get("task_support", {}).items():
        lines.append(
            "| {} | {} |".format(
                TASK_BY_ID[task_id]["name"],
                support_labels[level],
            )
        )
    lines.extend(
        [
        "",
        "## Attributes",
        "",
        "| Attribute or group | Type | Role | Description |",
        "|---|---|---|---|",
        ]
    )
    for attribute in metadata.get("attributes", []):
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                attribute["name"],
                attribute.get("dtype", "—"),
                attribute.get("role", "feature"),
                attribute["description"].replace("|", "\\|"),
            )
        )

    statistics = characteristics.get("sequence_statistics")
    if statistics:
        sequences = statistics["sequences"]
        lengths = [row["samples"] for row in sequences]
        total = sum(lengths)
        missing = sum(row["missing_rows"] for row in sequences)
        lines.extend([
            "", "## Sequence statistics", "", statistics["description"], "",
            "| Statistic | Value |", "|---|---:|",
            f"| Sequences | {len(sequences):,} |",
            f"| Total samples | {total:,} |",
            f"| Minimum sequence length | {min(lengths):,} |",
            f"| Maximum sequence length | {max(lengths):,} |",
            f"| Mean sequence length | {total / len(sequences):,.0f} |",
            f"| Rows with missing signals | {missing:,} |", "",
            "| Sequence ID | Samples | Rows with missing signals |",
            "|---|---:|---:|",
        ])
        for row in sequences:
            lines.append(f"| {row['id']} | {row['samples']:,} | {row['missing_rows']:,} |")
        lines.append(f"| **Total** | **{total:,}** | **{missing:,}** |")

    notes = metadata.get("notes", [])
    if notes:
        lines.extend(["", "## Usage notes", ""])
        lines.extend("- {}".format(note) for note in notes)

    download = metadata.get("download")
    if download:
        variants = download.get("variants", {})
        default_variant = download.get("default_variant")
        command = "uv run --locked python scripts/download.py {}".format(
            metadata["id"]
        )
        if default_variant:
            command += " --variant {}".format(default_variant)
        lines.extend(
            [
                "",
                "## Download",
                "",
                "```bash",
                command,
                "```",
                "",
            ]
        )
        if variants:
            lines.extend(
                [
                    "Available variants: {}.".format(
                        ", ".join("`{}`".format(item) for item in sorted(variants))
                    ),
                    "",
                    "Source page: {}".format(
                        _link(
                            source["name"],
                            source.get("landing_page", ""),
                        )
                    ),
                ]
            )
        elif download.get("method") == "files":
            lines.append("Source files (bundled locally without changing their contents):")
            lines.append("")
            lines.extend(
                "- " + _link(item["filename"], item["url"])
                for item in download["files"]
            )
        elif download.get("method", "url") == "kaggle":
            lines.append(
                "Kaggle dataset: [`{}`](https://www.kaggle.com/datasets/{})".format(
                    download["dataset"], download["dataset"]
                )
            )
        else:
            lines.append(
                "Source archive: [{}]({})".format(
                    download["filename"], download["url"]
                )
            )

    citation = metadata.get("citation")
    if citation:
        lines.extend(["", "## Suggested citation", "", citation])

    lines.extend(["", END])
    return "\n".join(lines)


def _replace_generated_section(text: str, generated: str) -> str:
    if BEGIN in text and END in text:
        before, remainder = text.split(BEGIN, 1)
        _, after = remainder.split(END, 1)
        return before.rstrip() + "\n\n" + generated + after

    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join([lines[0], "", generated, ""] + lines[1:]).rstrip() + "\n"
    return generated + "\n\n" + text.lstrip()


def update_dataset_readme(metadata: Dict[str, Any]) -> Path:
    """Insert or update the generated metadata section in one README."""
    directory = DATASETS_ROOT / metadata["id"]
    candidates = [directory / "README.md", directory / "readme.md"]
    readme = next((path for path in candidates if path.exists()), candidates[0])
    current = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if not current:
        current = "# {}\n".format(metadata["name"])
    updated = _replace_generated_section(current, render_metadata(metadata))
    readme.write_text(updated, encoding="utf-8")
    return readme


def update_all_readmes() -> List[Path]:
    """Synchronize every dataset README from the catalog."""
    return [update_dataset_readme(metadata) for metadata in load_catalog()]


def render_catalog_table(catalog: Iterable[Dict[str, Any]]) -> str:
    """Render the compact table used by the dataset catalog README."""
    lines = [
        "| Dataset | Data type | Tasks | License | Access |",
        "|---|---|---|---|---|",
    ]
    for item in catalog:
        if item["availability"]["status"] != "available":
            continue
        lines.append(
            "| [{short}](datasets/{id}/{readme}) | {kind} | "
            "{tasks} | {license} | {access} |".format(
                short=item["short_name"],
                id=item["id"],
                readme=item.get("readme", "README.md"),
                kind=item.get("characteristics", {}).get("kind", "—"),
                tasks=", ".join(item.get("tasks", [])),
                license=item["license"]["name"],
                access=item["availability"]["access"],
            )
        )
    return "\n".join(lines)


def render_task_table(catalog: Iterable[Dict[str, Any]]) -> str:
    """Render dataset suitability for canonical predictive-maintenance tasks."""
    lines = [
        "| Dataset | {} |".format(
            " | ".join(task["short_name"] for task in TASKS)
        ),
        "|---|{}|".format("|".join(":---:" for _ in TASKS)),
    ]
    symbols = {"direct": "✓", "derived": "△"}
    for item in catalog:
        support = item.get("task_support", {})
        cells = [symbols.get(support.get(task["id"]), "—") for task in TASKS]
        lines.append(
            "| [{short}](datasets/{id}/{readme}) | {cells} |".format(
                short=item["short_name"],
                id=item["id"],
                readme=item.get("readme", "README.md"),
                cells=" | ".join(cells),
            )
        )
    return "\n".join(lines)


# Taxonomy ids that have an implemented task package with a README.
_TASK_PACKAGE_BY_ID = {
    "rul_prediction": "rul",
    "time_to_event": "tte",
}


def render_task_definitions() -> str:
    """Render definitions for the canonical task taxonomy."""
    lines = []
    for task in TASKS:
        package = _TASK_PACKAGE_BY_ID.get(task["id"])
        if package is not None:
            name = "[{}](tasks/{}/README.md)".format(task["name"], package)
        else:
            name = task["name"]
        lines.append("- **{}:** {}".format(name, task["description"]))
    return "\n".join(lines)


def update_catalog_readme() -> Path:
    """Update generated catalog and task tables in the dataset catalog README."""
    readme = ROOT / "README.md"
    current = readme.read_text(encoding="utf-8")
    catalog_generated = "\n".join(
        [
            CATALOG_BEGIN,
            "",
            render_catalog_table(load_catalog()),
            "",
            CATALOG_END,
        ]
    )
    task_generated = "\n".join(
        [
            TASKS_BEGIN,
            "",
            render_task_definitions(),
            "",
            "The matrix distinguishes immediately usable targets from tasks that require",
            "label, endpoint, or health-index construction.",
            "",
            "- ✓: directly supported by labels, targets, or event/censoring records",
            "- △: supported after deriving labels or targets from chronology or domain assumptions",
            "- —: not a natural use of the dataset",
            "",
            render_task_table(load_catalog()),
            "",
            TASKS_END,
        ]
    )
    if CATALOG_BEGIN not in current or CATALOG_END not in current:
        raise ValueError(
            "{} does not contain dataset-table markers".format(readme)
        )
    before, remainder = current.split(CATALOG_BEGIN, 1)
    _, after = remainder.split(CATALOG_END, 1)
    current = before.rstrip() + "\n\n" + catalog_generated + after
    if TASKS_BEGIN not in current or TASKS_END not in current:
        raise ValueError("{} does not contain task-table markers".format(readme))
    before, remainder = current.split(TASKS_BEGIN, 1)
    _, after = remainder.split(TASKS_END, 1)
    readme.write_text(before.rstrip() + "\n\n" + task_generated + after, encoding="utf-8")
    return readme
