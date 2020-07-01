import abc
from dataclasses import dataclass
from typing import Any, Tuple, Iterator

from .dimension import Dimension, Scalar


class Scale(abc.ABC):
    @abc.abstractmethod
    def unit(self, dimension: Dimension) -> Tuple["Quantity", str]: ...


@dataclass(frozen=True, eq=False, order=False)
class Quantity:
    value: Any
    dimension: Dimension
    scale: Scale

    def set_scale(self, curr: "Quantity", new: "Quantity") -> "Quantity":
        if self.scale != curr.scale:
            raise ValueError("Error setting scale: Can't set scale for {} with {}".format(self, curr))
        if curr.dimension != new.dimension:
            raise ValueError("Error setting scale: {} and {} have different mass dimensions".format(curr, new))
        if curr.scale == new.scale:
            raise ValueError("Error setting scale: Initial and final scale are the same")
        return Quantity(
            self.value * self.dimension.scale(
                (new.value / curr.value)**(1 / curr.dimension.mass_dim)
            ),
            self.dimension,
            new.scale,
        )

    def in_unit(self, val: "Quantity") -> Any:
        try:
            res = self / val
        except ValueError:
            raise ValueError("Can't convert units: incompatible scales")
        if res.dimension.mass_dim != 0:
            raise ValueError(
                "Can't convert units: incompatible mass dimensions {} and {}"
                .format(self.dimension.mass_dim, val.dimension.mass_dim)
            )
        return res.value

    def __str__(self) -> str:
        unit, suffix = self.scale.unit(self.dimension)
        if suffix != "":
            return str(self.in_unit(unit)) + " " + suffix
        else:
            return str(self.in_unit(unit))

    def __format__(self, format_str: str) -> str:
        unit, suffix = self.scale.unit(self.dimension)
        if suffix != "":
            return ("{:" + format_str + "} {}").format(
                self.in_unit(unit),
                suffix,
            )
        else:
            return ("{:" + format_str + "}").format(
                self.in_unit(unit),
            )

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, key: Any) -> "Quantity":
        return Quantity(
            self.value[key],
            self.dimension,
            self.scale,
        )

    def __iter__(self) -> "Iterator[Quantity]":
        for value in iter(self.value):
            yield Quantity(
                value,
                self.dimension,
                self.scale,
            )

    def __reversed__(self) -> "Iterator[Quantity]":
        for value in reversed(self.value):
            yield Quantity(
                value,
                self.dimension,
                self.scale,
            )

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(
                    "Can't compare values: incompatible mass dimensions {} and {}"
                    .format(self.dimension.mass_dim, other.dimension.mass_dim)
                )

            if self.dimension != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value == other.value
        else:
            if self.dimension != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.dimension.mass_dim))

            return self.value == other

    def __lt__(self, other: Any) -> Any:
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(
                    "Can't compare values: incompatible mass dimensions {} and {}"
                    .format(self.dimension.mass_dim, other.dimension.mass_dim)
                )

            if self.dimension != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value < other.value
        else:
            if self.dimension != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.dimension.mass_dim))

            return self.value < other

    def __le__(self, other: Any) -> Any:
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(
                    "Can't compare values: incompatible mass dimensions {} and {}"
                    .format(self.dimension.mass_dim, other.dimension.mass_dim)
                )

            if self.dimension != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value <= other.value
        else:
            if self.dimension != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.dimension.mass_dim))

            return self.value <= other

    def __gt__(self, other: Any) -> Any:
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(
                    "Can't compare values: incompatible mass dimensions {} and {}"
                    .format(self.dimension.mass_dim, other.dimension.mass_dim)
                )

            if self.dimension != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value > other.value
        else:
            if self.dimension != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.dimension.mass_dim))

            return self.value > other

    def __ge__(self, other: Any) -> Any:
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(
                    "Can't compare values: incompatible mass dimensions {} and {}"
                    .format(self.dimension.mass_dim, other.dimension.mass_dim)
                )

            if self.dimension != Scalar and self.scale != other.scale:
                raise ValueError("Can't compare values: incompatible scales")

            return self.value >= other.value
        else:
            if self.dimension != Scalar:
                raise ValueError("Can't compare scalar to mass dimension {}".format(self.dimension.mass_dim))

            return self.value >= other

    def __neg__(self) -> "Quantity":
        return Quantity(
            -self.value,
            self.dimension,
            self.scale,
        )

    def __add__(self, other: Any) -> "Quantity":
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(
                    "Can't add values: incompatible mass dimensions {} and {}"
                    .format(self.dimension.mass_dim, other.dimension.mass_dim)
                )

            if self.dimension != Scalar and self.scale != other.scale:
                raise ValueError("Can't add values: incompatible scales")

            return Quantity(
                self.value + other.value,
                self.dimension,
                self.scale,
            )
        else:
            if self.dimension != Scalar:
                raise ValueError("Can't add scalar to mass dimension {}".format(self.dimension.mass_dim))

            return Quantity(
                self.value + other,
                self.dimension,
                self.scale,
            )

    def __radd__(self, other: Any) -> "Quantity":
        if self.dimension != Scalar:
            raise ValueError("Can't add mass dimension {} to scalar".format(self.dimension.mass_dim))

        return Quantity(
            other + self.value,
            self.dimension,
            self.scale,
        )

    def __sub__(self, other: Any) -> "Quantity":
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(
                    "Can't subtract values: incompatible mass dimensions {} and {}"
                    .format(self.dimension.mass_dim, other.dimension.mass_dim)
                )

            if self.dimension != Scalar and self.scale != other.scale:
                raise ValueError("Can't subtract values: incompatible scales")

            return Quantity(
                self.value - other.value,
                self.dimension,
                self.scale,
            )
        else:
            if self.dimension != Scalar:
                raise ValueError("Can't subtract scalar from mass dimension {}".format(self.dimension.mass_dim))

            return Quantity(
                self.value - other,
                self.dimension,
                self.scale,
            )

    def __rsub__(self, other: Any) -> "Quantity":
        if self.dimension != Scalar:
            raise ValueError("Can't subtract mass dimension {} from scalar".format(self.dimension.mass_dim))

        return Quantity(
            other - self.value,
            self.dimension,
            self.scale,
        )

    def __mul__(self, other: Any) -> "Quantity":
        if isinstance(other, Quantity):
            if other.dimension == Scalar:
                scale = self.scale
            elif self.dimension == Scalar:
                scale = other.scale
            elif self.scale == other.scale:
                scale = self.scale
            else:
                raise ValueError("Can't multiply values: incompatible scales")

            return Quantity(
                self.value * other.value,
                self.dimension * other.dimension,
                scale,
            )
        else:
            return Quantity(
                self.value * other,
                self.dimension,
                self.scale,
            )

    def __rmul__(self, other: Any) -> "Quantity":
        return Quantity(
            other * self.value,
            self.dimension,
            self.scale,
        )

    def __truediv__(self, other: Any) -> "Quantity":
        if isinstance(other, Quantity):
            if other.dimension == Scalar:
                scale = self.scale
            elif self.dimension == Scalar:
                scale = other.scale
            elif self.scale == other.scale:
                scale = self.scale
            else:
                raise ValueError("Can't divide values: incompatible scales")

            return Quantity(
                self.value / other.value,
                self.dimension / other.dimension,
                scale,
            )
        else:
            return Quantity(
                self.value / other,
                self.dimension,
                self.scale,
            )

    def __rtruediv__(self, other: Any) -> "Quantity":
        return Quantity(
            other / self.value,
            Scalar / self.dimension,
            self.scale,
        )

    def __pow__(self, other: int) -> "Quantity":
        return Quantity(
            self.value**other,
            self.dimension**other,
            self.scale,
        )

    def root(self, other: int) -> "Quantity":
        return Quantity(
            self.value**(1 / other),
            self.dimension.root(other),
            self.scale,
        )

    def sqrt(self) -> "Quantity":
        return self.root(2)

    __array_ufunc__ = None
    """Ensures proper numpy integration by preventing numpy from
    distributing across the numpy array before the value is unwrapped by
    rmul. Only works for numpy >= 1.13"""
