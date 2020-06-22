from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
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

    def __mul__(self, other: "Unit") -> "Unit":
        return Unit(self.mass_dim + other.mass_dim)

    def __truediv__(self, other: "Unit") -> "Unit":
        return Unit(self.mass_dim - other.mass_dim)

    def __pow__(self, exponent: int) -> "Unit":
        return Unit(self.mass_dim * exponent)

    def root(self, exponent: int) -> "Unit":
        if self.mass_dim % exponent != 0:
            if exponent == 2:
                raise ValueError("Can't take square root of {}".format(self))
            if exponent == 3:
                raise ValueError("Can't take cube root of {}".format(self))
            raise ValueError("Can't take {}-th root of {}".format(exponent, self))
        return Unit(self.mass_dim // exponent)

    def sqrt(self) -> "Unit":
        return self.root(2)

    def scale(self, scale: float) -> float:
        return scale**self.mass_dim


Scalar = Unit(0)
Mass = Unit(1)
Length = Unit(-1)
