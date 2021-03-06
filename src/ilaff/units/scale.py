from dataclasses import dataclass
from typing import Tuple

from .dimension import Dimension, Mass, Length, Scalar
from .quantity import Scale, Quantity


@dataclass(frozen=True)
class Physical(Scale):
    def unit(self, dimension: Dimension) -> Tuple[Quantity, str]:
        if dimension.mass_dim == 1:
            return (GeV, "GeV")
        if dimension.mass_dim == -1:
            return (fm, "fm")
        if dimension.mass_dim > 0:
            return (GeV**dimension.mass_dim, "GeV^{}".format(dimension.mass_dim))
        if dimension.mass_dim < 0:
            return (fm**(-dimension.mass_dim), "fm^{}".format(-dimension.mass_dim))
        return (one, "")


@dataclass(frozen=True, eq=False)
class Lattice(Scale):
    def unit(self, dimension: Dimension) -> Tuple[Quantity, str]:
        if dimension.mass_dim == -1:
            return (a(self), "a")
        if dimension.mass_dim != 0:
            return (a(self)**(-dimension.mass_dim), "a^{}".format(-dimension.mass_dim))
        return (one, "")


MeV = Quantity(0.001, Mass, Physical())
GeV = Quantity(1.0, Mass, Physical())
fm = Quantity(1.0 / 0.1973269788, Length, Physical())
one = Quantity(1.0, Scalar, Physical())


def a(scale: Scale) -> Quantity:
    return Quantity(1.0, Length, scale)
