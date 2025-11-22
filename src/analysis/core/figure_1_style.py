"""Style helpers for Figure 1 plotting."""

from __future__ import annotations

from typing import Tuple

from ..plotting import (
    a4_size_inches as base_a4_size_inches,
    apply_paper_font_style,
)


def set_global_fonts(font_family: str = "Times New Roman", font_size: int = 12) -> None:
    """Apply Times-like fonts to matplotlib."""
    apply_paper_font_style(
        font_family=font_family,
        font_size=font_size,
        mathtext_fontset="dejavuserif",
    )


def a4_size_inches(orientation: str = "landscape") -> Tuple[float, float]:
    """Return (width, height) for an A4 page in the requested orientation."""
    return base_a4_size_inches(orientation)


def lighten_hex(hex_color: str, factor: float = 0.65) -> str:
    """Return a lighter hex color by blending with white."""
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    red = 1 - (1 - red) * factor
    green = 1 - (1 - green) * factor
    blue = 1 - (1 - blue) * factor
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


__all__ = ["set_global_fonts", "a4_size_inches", "lighten_hex"]
