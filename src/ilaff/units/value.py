import abc
import attr
from typing import Any, Optional, Tuple

from .unit import Unit, Scalar


class Scale(abc.ABC):
    def shared(self, other: "Scale") -> "Optional[Scale]":
        if self == other:
            return self
        return None

    def to(self, other: "Scale", unit: Unit) -> float:
        if self == other:
            return 1.0
        raise ValueError("Can't scale {} to {}".format(self, other))

    @abc.abstractmethod
    def unit(self, unit: Unit) -> Tuple[float, str]:
        raise NotImplementedError


@attr.s(frozen=True, auto_attribs=True)
class Value:
    value: Any
    unit: Unit
    scale: Scale

    def in_units(self, scale: Scale) -> "Value":
        if self.scale == scale:
            return self
        else:
            return Value(
                self.value * self.scale.to(scale, self.unit),
                self.unit,
                scale,
            )

    def __str__(self) -> str:
        scale, unit = self.scale.unit(self.unit)
        if unit != "":
            return str(self.value * scale) + " " + unit
        else:
            return str(self.value * scale)

    def __format__(self, format_str: str) -> str:
        scale, unit = self.scale.unit(self.unit)
        if unit != "":
            return (self.value * scale).__format__(format_str) + " " + unit
        else:
            return (self.value * scale).__format__(format_str)

    def __neg__(self) -> "Value":
        return Value(
            -self.value,
            self.unit,
            self.scale,
        )

    def __add__(self, other: Any) -> "Value":
        if isinstance(other, Value):
            if self.unit != other.unit:
                raise ValueError("Can't add values: incompatible mass dimensions {} and {}".format(self.unit.mass_dim, other.unit.mass_dim))

            if self.unit != Scalar and self.scale != other.scale:
                raise ValueError("Can't add values: incompatible scales")

            return Value(
                self.value + other.value,
                self.unit,
                self.scale,
            )
        else:
            if self.unit != Scalar:
                raise ValueError("Can't add scalar to mass dimension {}".format(self.unit.mass_dim))

            return Value(
                self.value + other,
                self.unit,
                self.scale,
            )

    def __radd__(self, other: Any) -> "Value":
        if self.unit != Scalar:
            raise ValueError("Can't add mass dimension {} to scalar".format(self.unit.mass_dim))

        return Value(
            other + self.value,
            self.unit,
            self.scale,
        )

    def __sub__(self, other: Any) -> "Value":
        if isinstance(other, Value):
            if self.unit != other.unit:
                raise ValueError("Can't subtract values: incompatible mass dimensions {} and {}".format(self.unit.mass_dim, other.unit.mass_dim))

            if self.unit != Scalar and self.scale != other.scale:
                raise ValueError("Can't subtract values: incompatible scales")

            return Value(
                self.value - other.value,
                self.unit,
                self.scale,
            )
        else:
            if self.unit != Scalar:
                raise ValueError("Can't subtract scalar from mass dimension {}".format(self.unit.mass_dim))

            return Value(
                self.value - other,
                self.unit,
                self.scale,
            )

    def __rsub__(self, other: Any) -> "Value":
        if self.unit != Scalar:
            raise ValueError("Can't subtract mass dimension {} from scalar".format(self.unit.mass_dim))

        return Value(
            other - self.value,
            self.unit,
            self.scale,
        )

    def __mul__(self, other: Any) -> "Value":
        if isinstance(other, Value):
            if other.unit == Scalar:
                scale = self.scale
            elif self.unit == Scalar:
                scale = other.scale
            elif self.scale == other.scale:
                scale = self.scale
            else:
                raise ValueError("Can't multiply values: incompatible scales")

            return Value(
                self.value * other.value,
                self.unit * other.unit,
                scale,
            )
        else:
            return Value(
                self.value * other,
                self.unit,
                self.scale,
            )

    def __rmul__(self, other: Any) -> "Value":
        return Value(
            other * self.value,
            self.unit,
            self.scale,
        )

    def __truediv__(self, other: Any) -> "Value":
        if isinstance(other, Value):
            if other.unit == Scalar:
                scale = self.scale
            elif self.unit == Scalar:
                scale = other.scale
            elif self.scale == other.scale:
                scale = self.scale
            else:
                raise ValueError("Can't divide values: incompatible scales")

            return Value(
                self.value / other.value,
                self.unit / other.unit,
                scale,
            )
        else:
            return Value(
                self.value / other,
                self.unit,
                self.scale,
            )

    def __rtruediv__(self, other: Any) -> "Value":
        return Value(
            other / self.value,
            Scalar / self.unit,
            self.scale,
        )

    def __pow__(self, other: int) -> "Value":
        return Value(
            self.value**other,
            self.unit**other,
            self.scale,
        )

    def root(self, other: int) -> "Value":
        return Value(
            self.value**(1 / other),
            self.unit.root(other),
            self.scale,
        )

    def sqrt(self) -> "Value":
        return self.root(2)
