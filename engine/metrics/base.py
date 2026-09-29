"""The value type every metric function returns."""

from dataclasses import dataclass


@dataclass
class MV:
    """One metric value.

    Attributes:
        value: the metric's value (a float, or 0.0 / 1.0 for a flag)
        worst: the bad-tail window value, for windowed metrics
        note: a short human-readable reason (for flags)
    """

    value: float
    worst: "float | None" = None
    note: "str | None" = None
