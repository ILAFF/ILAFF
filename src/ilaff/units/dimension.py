from dataclasses import dataclass


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

    def __pow__(self, exponent: int) -> "Dimension":
        return Dimension(self.mass_dim * exponent)

    def root(self, exponent: int) -> "Dimension":
        if self.mass_dim % exponent != 0:
            if exponent == 2:
                raise ValueError("Can't take square root of {}".format(self))
            if exponent == 3:
                raise ValueError("Can't take cube root of {}".format(self))
            raise ValueError("Can't take {}-th root of {}".format(exponent, self))
        return Dimension(self.mass_dim // exponent)

    def sqrt(self) -> "Dimension":
        return self.root(2)

    def scale(self, scale: float) -> float:
        return scale**self.mass_dim


Scalar = Dimension(0)
Mass = Dimension(1)
Length = Dimension(-1)
