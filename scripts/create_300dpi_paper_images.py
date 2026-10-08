from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "results"
OUTPUT_ROOT = SOURCE_ROOT / "paper_300dpi_images"
MANIFEST_PATH = OUTPUT_ROOT / "paper_300dpi_image_manifest.csv"

TARGET_DPI = 300
MIN_WIDTH_PX = 1800
MIN_HEIGHT_PX = 1200


def iter_source_images() -> list[Path]:
    images: list[Path] = []
    for path in SOURCE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if OUTPUT_ROOT in path.parents:
            continue
        if path.suffix.lower() in {".png", ".svg"}:
            images.append(path)
    return sorted(images)


def scale_for_paper(width: int, height: int) -> float:
    scale = 1.0
    if width < MIN_WIDTH_PX:
        scale = max(scale, MIN_WIDTH_PX / width)
    if height < MIN_HEIGHT_PX:
        scale = max(scale, MIN_HEIGHT_PX / height)
    return scale


def save_png_300dpi(source: Path, destination: Path) -> dict[str, object]:
    with Image.open(source) as image:
        original_width, original_height = image.size
        scale = scale_for_paper(original_width, original_height)

        if scale > 1.0:
            new_size = (
                int(round(original_width * scale)),
                int(round(original_height * scale)),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            new_size = image.size

        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, dpi=(TARGET_DPI, TARGET_DPI), optimize=True)

    return {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "output": str(destination.relative_to(PROJECT_ROOT)),
        "kind": "png_300dpi",
        "original_width_px": original_width,
        "original_height_px": original_height,
        "output_width_px": new_size[0],
        "output_height_px": new_size[1],
        "scale": round(scale, 4),
        "dpi": TARGET_DPI,
    }


def copy_vector(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "output": str(destination.relative_to(PROJECT_ROOT)),
        "kind": "svg_vector_copy",
        "original_width_px": "",
        "original_height_px": "",
        "output_width_px": "",
        "output_height_px": "",
        "scale": "",
        "dpi": "vector",
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for source in iter_source_images():
        relative = source.relative_to(SOURCE_ROOT)
        destination = OUTPUT_ROOT / relative
        if source.suffix.lower() == ".png":
            rows.append(save_png_300dpi(source, destination))
        elif source.suffix.lower() == ".svg":
            rows.append(copy_vector(source, destination))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "output",
        "kind",
        "original_width_px",
        "original_height_px",
        "output_width_px",
        "output_height_px",
        "scale",
        "dpi",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    png_count = sum(row["kind"] == "png_300dpi" for row in rows)
    svg_count = sum(row["kind"] == "svg_vector_copy" for row in rows)
    print(f"Created {png_count} PNG 300 DPI images and copied {svg_count} SVG files.")
    print(f"Output folder: {OUTPUT_ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
