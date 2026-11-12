"""Presentation precision; calculation records always retain their numeric values."""

THICKNESS_DECIMALS = 2


def format_thickness_mm(value, missing="未评定"):
    return missing if value is None else f"{value:.{THICKNESS_DECIMALS}f}"


def thickness_display_data(value):
    """Format a UI copy of structured tool output without changing raw evidence."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            thickness = key in {"thickness_mm", "thickness_mean_mm", "mean_thickness_mm", "h_mm"}
            thickness |= key == "value_mm" and "d1" in value and "d2" in value
            result[key] = format_thickness_mm(item) if thickness and isinstance(item, (int, float)) else thickness_display_data(item)
        return result
    if isinstance(value, (list, tuple)):
        return [thickness_display_data(item) for item in value]
    return value
