"""Generate human-readable documentation from dataset metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .catalog import ROOT, load_catalog


BEGIN = "<!-- BEGIN GENERATED METADATA -->"
END = "<!-- END GENERATED METADATA -->"
CATALOG_BEGIN = "<!-- BEGIN GENERATED DATASET TABLE -->"
CATALOG_END = "<!-- END GENERATED DATASET TABLE -->"


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
        "> This section is generated from `metadata.json`. Do not edit it directly.",
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
            _link(source.get("doi", "—"), source.get("doi_url", ""))
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
        "| Tasks | {} |".format(", ".join(metadata.get("tasks", []))),
        "",
        "## Attributes",
        "",
        "| Attribute or group | Type | Role | Description |",
        "|---|---|---|---|",
    ]
    for attribute in metadata.get("attributes", []):
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                attribute["name"],
                attribute.get("dtype", "—"),
                attribute.get("role", "feature"),
                attribute["description"].replace("|", "\\|"),
            )
        )

    notes = metadata.get("notes", [])
    if notes:
        lines.extend(["", "## Usage notes", ""])
        lines.extend("- {}".format(note) for note in notes)

    download = metadata.get("download")
    if download:
        lines.extend(
            [
                "",
                "## Download",
                "",
                "```bash",
                "python scripts/download.py {}".format(metadata["id"]),
                "```",
                "",
            ]
        )
        if download.get("method", "url") == "kaggle":
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
    directory = ROOT / metadata["id"]
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
    """Render the compact table used by the repository README."""
    lines = [
        "| Dataset | Available | Data type | Tasks | License | Access |",
        "|---|:---:|---|---|---|---|",
    ]
    for item in catalog:
        lines.append(
            "| [{short}](datasets/{id}/{readme}) | {available} | {kind} | "
            "{tasks} | {license} | {access} |".format(
                short=item["short_name"],
                id=item["id"],
                readme=item.get("readme", "README.md"),
                available="✓"
                if item["availability"]["status"] == "available"
                else "—",
                kind=item.get("characteristics", {}).get("kind", "—"),
                tasks=", ".join(item.get("tasks", [])),
                license=item["license"]["name"],
                access=item["availability"]["access"],
            )
        )
    return "\n".join(lines)


def update_repository_readme() -> Path:
    """Update the generated catalog table in the repository README."""
    readme = ROOT.parent / "README.md"
    current = readme.read_text(encoding="utf-8")
    generated = "\n".join(
        [
            CATALOG_BEGIN,
            "",
            render_catalog_table(load_catalog()),
            "",
            CATALOG_END,
        ]
    )
    if CATALOG_BEGIN not in current or CATALOG_END not in current:
        raise ValueError(
            "{} does not contain dataset-table markers".format(readme)
        )
    before, remainder = current.split(CATALOG_BEGIN, 1)
    _, after = remainder.split(CATALOG_END, 1)
    readme.write_text(
        before.rstrip() + "\n\n" + generated + after,
        encoding="utf-8",
    )
    return readme
