"""v0.2 gate. Not calculated in v0.1."""

from excavation_lab.analytics.models import EventRow, TruckVisitRow
from excavation_lab.errors import LabError

__all__ = ["EventRow", "TruckVisitRow", "unavailable_error"]


def unavailable_error() -> LabError:
    return LabError(
        code="unavailable",
        message="События и показатели в версии 0.1 не рассчитываются.",
    )
