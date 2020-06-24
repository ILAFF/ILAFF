from dataclasses import dataclass
from typing import Tuple

from .unit import Unit, Mass, Length, Scalar
from .value import Scale, Value


@dataclass(frozen=True)
class Physical(Scale):
    def unit(self, unit: Unit) -> Tuple[Value, str]:
        if unit.mass_dim == 1:
            return (GeV, "GeV")
        if unit.mass_dim == -1:
            return (fm, "fm")
        if unit.mass_dim > 0:
            return (GeV**unit.mass_dim, "GeV^{}".format(unit.mass_dim))
        if unit.mass_dim < 0:
            return (fm**(-unit.mass_dim), "fm^{}".format(-unit.mass_dim))
        return (one, "")


@dataclass(frozen=True, eq=False)
class Lattice(Scale):
    def unit(self, unit: Unit) -> Tuple[Value, str]:
        if unit.mass_dim == -1:
            return (a(self), "a")
        if unit.mass_dim != 0:
            return (a(self)**(-unit.mass_dim), "a^{}".format(-unit.mass_dim))
        return (one, "")


MeV = Value(0.001, Mass, Physical())
GeV = Value(1.0, Mass, Physical())
fm = Value(1.0 / 0.1973269788, Length, Physical())
one = Value(1.0, Scalar, Physical())


def a(scale: Scale) -> Value:
    return Value(1.0, Length, scale)
