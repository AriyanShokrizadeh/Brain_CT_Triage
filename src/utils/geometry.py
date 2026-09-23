"""Geometry utilities."""

from math import dist


def rescaled_spacing(
    original_shape: tuple[int, int],
    original_spacing: tuple[float, float],
    new_shape: tuple[int, int],
) -> tuple[float, float]:
    """Return pixel spacing after image resizing."""
    original_height, original_width = original_shape
    new_height, new_width = new_shape
    spacing_x, spacing_y = original_spacing

    new_spacing_x = spacing_x * original_width / new_width
    new_spacing_y = spacing_y * original_height / new_height

    return new_spacing_x, new_spacing_y


def point_to_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float | None:
    """Return perpendicular distance from a point to a line."""
    px, py = point
    ax, ay = start
    bx, by = end

    line_length = dist(start, end)

    if line_length <= 1e-8:
        return None

    numerator = abs((bx - ax) * (ay - py) - (ax - px) * (by - ay))

    return numerator / line_length
