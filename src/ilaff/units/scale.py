import attr
from typing import Tuple, Optional, MutableMapping

from .unit import Unit
from .value import Scale, Value


@attr.s(frozen=True, auto_attribs=True)
class Physical(Scale):
    def shared(self, other: Scale) -> Optional[Scale]:
        if self == other:
            return other
        if other.shared(self) == self:
            return self
        return None

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


@attr.s(frozen=True, auto_attribs=True, cmp=False)
class Lattice(Scale):
    scale: MutableMapping[Scale, float] = attr.Factory(dict)

    def shared(self, other: Scale) -> Optional[Scale]:
        if self == other:
            return self
        if other in self.scale:
            return other
        if isinstance(other, Lattice):
            scales = [scale for scale in self.scale.keys() if scale in other.scale and not isinstance(scale, Lattice)]
            if len(scales) == 1:
                return scales[0]
        return None

    def to(self, other: Scale, unit: Unit) -> float:
        if self == other:
            return 1.0
        if other in self.scale:
            return unit.scale(self.scale[other])
        raise ValueError("Can't scale {} to {}".format(self, other))

    def set_scale(self, value: Value, other: Value):
        if value.scale != self:
            raise ValueError("Error setting scale: Can't set scale for {} with {}".format(self, value))
        if value.unit != other.unit:
            raise ValueError("Error setting scale: {} and {} have different mass dimensions".format(value, other))
        self.scale[other.scale] = (other.value / value.value)**(1 / value.unit.mass_dim)

    def unit(self, unit: Unit) -> Tuple[float, str]:
        if unit.mass_dim == -1:
            return (1.0, "a")
        if unit.mass_dim != 0:
            return (1.0, "a^{}".format(-unit.mass_dim))
        return (1.0, "")
