from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = (
    ROOT
    / "results"
    / "01_fpt_reference"
    / "task1_fpt_visualizations"
    / "task1_fpt_overview.pdf"
)
OUTPUT_SVG = (
    ROOT
    / "figures"
    / "fpt_reference_irrms_overview.svg"
)

OLD_LABEL = '<tspan y="0" x="0 8.272">HI</tspan>'
NEW_LABEL = '<tspan y="0" x="-10">IRRMS</tspan>'
ORIGINAL_ROOT = (
    'width="1070.6206" height="754.42129" '
    'viewBox="0 0 1070.6206 754.42129"'
)
CROPPED_ROOT = (
    'width="1070.6206" height="699.42129" '
    'viewBox="0 55 1070.6206 699.42129"'
)
SOURCE_FONT = 'font-family="DejaVuSans"'
PAPER_FONT = 'font-family="Times New Roman"'


def main() -> None:
    with fitz.open(SOURCE_PDF) as document:
        if document.page_count != 1:
            raise ValueError("Expected a one-page overview PDF")
        svg = document[0].get_svg_image(text_as_path=False)

    label_count = svg.count(OLD_LABEL)
    if label_count != 6:
        raise ValueError(f"Expected six HI labels, found {label_count}")

    svg = svg.replace(OLD_LABEL, NEW_LABEL)
    if svg.count(ORIGINAL_ROOT) != 1:
        raise ValueError("Unexpected SVG page geometry")
    svg = svg.replace(ORIGINAL_ROOT, CROPPED_ROOT)
    if SOURCE_FONT not in svg:
        raise ValueError("Expected DejaVuSans text in source SVG")
    svg = svg.replace(SOURCE_FONT, PAPER_FONT)
    if "<image" in svg or "data:image" in svg:
        raise ValueError("Output contains an embedded raster image")
    new_count = svg.count(NEW_LABEL)
    remaining_old_count = svg.count(OLD_LABEL)
    if new_count != 6 or remaining_old_count != 0:
        raise ValueError(
            "Y-axis label replacement failed: "
            f"new={new_count}, remaining_old={remaining_old_count}"
        )
    if SOURCE_FONT in svg or PAPER_FONT not in svg:
        raise ValueError("Font replacement failed")

    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
