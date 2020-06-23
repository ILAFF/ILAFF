import abc
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .unit import Unit, Scalar


class Scale(abc.ABC):
    @abc.abstractmethod
    def unit(self, unit: Unit) -> Tuple["Value", str]:
        raise NotImplementedError


@dataclass(frozen=True, eq=False, order=False)
class Value:
    value: Any
    unit: Unit
    scale: Scale

    def set_scale(self, curr: "Value", new: "Value") -> "Value":
        if self.scale == new.scale:
            return self
        else:
            if self.scale != curr.scale:
                raise ValueError("Error setting scale: Can't set scale for {} with {}".format(self, curr))
            if curr.unit != new.unit:
                raise ValueError("Error setting scale: {} and {} have different mass dimensions".format(curr, new))
            return Value(
                self.value * self.unit.scale(
                    (new.value / curr.value)**(1 / curr.unit.mass_dim)
                ),
                self.unit,
                new.scale,
            )

    def in_unit(self, val: "Value") -> Any:
        try:
            res = self / val
        except ValueError:
            raise ValueError("Can't convert units: incompatible scales")
        if res.unit.mass_dim != 0:
            raise ValueError("Can't convert units: incompatible mass dimensions {} and {}".format(self.unit.mass_dim, val.unit.mass_dim))
        return res.value

    def __str__(self) -> str:
        unit, suffix = self.scale.unit(self.unit)
        if suffix != "":
            return str(self.in_unit(unit)) + " " + suffix
        else:
            return str(self.in_unit(unit))

    def __format__(self, format_str: str) -> str:
        unit, suffix = self.scale.unit(self.unit)
        if unit != "":
            return self.in_unit(unit).__format__(format_str) + " " + suffix
        else:
            return self.in_unit(unit).__format__(format_str)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Value):
            if self.unit != other.unit:
                raise ValueError("Can't compare values: incompatible mass dimensions {} and {}".format(self.unit.mass_dim, other.unit.mass_dim))

            if self.unit != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value == other.value
        else:
            if self.unit != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.unit.mass_dim))

            return self.value == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Value):
            if self.unit != other.unit:
                raise ValueError("Can't compare values: incompatible mass dimensions {} and {}".format(self.unit.mass_dim, other.unit.mass_dim))

            if self.unit != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value < other.value
        else:
            if self.unit != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.unit.mass_dim))

            return self.value < other

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Value):
            if self.unit != other.unit:
                raise ValueError("Can't compare values: incompatible mass dimensions {} and {}".format(self.unit.mass_dim, other.unit.mass_dim))

            if self.unit != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value <= other.value
        else:
            if self.unit != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.unit.mass_dim))

            return self.value <= other

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Value):
            if self.unit != other.unit:
                raise ValueError("Can't compare values: incompatible mass dimensions {} and {}".format(self.unit.mass_dim, other.unit.mass_dim))

            if self.unit != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value > other.value
        else:
            if self.unit != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.unit.mass_dim))

            return self.value > other

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, Value):
            if self.unit != other.unit:
                raise ValueError("Can't compare values: incompatible mass dimensions {} and {}".format(self.unit.mass_dim, other.unit.mass_dim))

            if self.unit != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value >= other.value
        else:
            if self.unit != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.unit.mass_dim))

            return self.value >= other

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

    __array_ufunc__ = None
    """Ensures proper numpy integration by preventing numpy from
    distributing across the numpy array before the value is unwrapped by
    rmul. Only works for numpy >= 1.13"""
