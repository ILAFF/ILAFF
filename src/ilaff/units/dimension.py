from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Dimension:
    mass_dim: int

    def __str__(self) -> str:
        if self.mass_dim == 1:
            return "mass"
        if self.mass_dim == -1:
            return "length"
        if self.mass_dim > 0:
            return "mass^{}".format(self.mass_dim)
        if self.mass_dim < 0:
            return "length^{}".format(-self.mass_dim)
        return "scalar"

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.mass_dim + other.mass_dim)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.mass_dim - other.mass_dim)

    def __pow__(self, exponent: Union[int, float]) -> "Dimension":
        dim = self.mass_dim * exponent
        if int(dim) != dim:
            if exponent == 1/2 and self.mass_dim % 2 != 0:
                raise ValueError("Can't take square root of {}".format(self))
            if exponent == 1/2 and self.mass_dim % 3 != 0:
                raise ValueError("Can't take cube root of {}".format(self))
            if (1 / exponent).is_integer():
                raise ValueError("Can't take {}-th root of {}".format(int(1/exponent), self))
            raise ValueError(
                "Can't raise {} to the power of {}: resulting mass dim {} is not an integer"
                .format(self, exponent, dim)
            )
        return Dimension(int(dim))

    def scale(self, scale: float) -> float:
        return scale**self.mass_dim


Scalar = Dimension(0)
Mass = Dimension(1)
Length = Dimension(-1)
