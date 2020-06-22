from dataclasses import dataclass, field
from typing import Tuple, Optional, MutableMapping

from .unit import Unit
from .value import Scale, Value


@dataclass(frozen=True)
class Physical(Scale):
    def unit(self, unit: Unit) -> Tuple[float, str]:
        if unit.mass_dim == 1:
            return (1000.0, "MeV")
        if unit.mass_dim == -1:
            return (0.1973269788, "fm")
        if unit.mass_dim > 0:
            return (1.0, "GeV^{}".format(unit.mass_dim))
        if unit.mass_dim < 0:
            return (1.0 / 0.1973269788**unit.mass_dim, "fm^{}".format(-unit.mass_dim))
        return (1.0, "")


@dataclass(frozen=True, eq=False)
class Lattice(Scale):
    def unit(self, unit: Unit) -> Tuple[float, str]:
        if unit.mass_dim == -1:
            return (1.0, "a")
        if unit.mass_dim != 0:
            return (1.0, "a^{}".format(-unit.mass_dim))
        return (1.0, "")
