"""Utilities for saving preprocessing comparison images."""

from pathlib import Path

from PIL import Image, ImageDraw


def save_comparison(
    original: Image.Image,
    box: tuple[int, int, int, int] | None,
    masked: Image.Image | None,
    save_path: Path,
) -> None:
    """Save the original and masked images side by side."""
    original_panel = original.convert("RGB")
    if box is not None:
        ImageDraw.Draw(original_panel).rectangle(box, outline="red", width=4)

    if masked is None:
        masked_panel = Image.new("RGB", original_panel.size, "black")
        ImageDraw.Draw(masked_panel).text((10, 10), "FAILED", fill="red")
    else:
        masked_panel = masked.convert("RGB")

    comparison = Image.new(
        "RGB",
        (
            original_panel.width + masked_panel.width,
            max(original_panel.height, masked_panel.height),
        ),
        "black",
    )
    comparison.paste(original_panel, (0, 0))
    comparison.paste(masked_panel, (original_panel.width, 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(save_path)
