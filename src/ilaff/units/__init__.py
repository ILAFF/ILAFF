from .scale import Physical, Lattice
from .unit import Mass, Length, Scalar
from .value import Value, Scale

MeV = Value(0.001, Mass, Physical())
GeV = Value(1.0, Mass, Physical())
fm = Value(1.0 / 0.1973269788, Length, Physical())


def a(scale: Scale) -> Value:
    return Value(1.0, Length, scale)
