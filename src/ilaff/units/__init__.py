"""Dimensional analysis for use in lattice QCD analyses.

A seperate instance of Lattice should be constructed for each independent
lattice spacing. This ensures that the independent lattice spacings are kept
distinct.

New values should generally be constructed by multiplying a float, numpy array,
or other raw value by one of the provided unit values (e.g. units.MeV,
units.GeV, units.fm, or units.a(scale)).

Examples:
    >>> str(0.135 * MeV)
    '0.135 MeV'
    >>> str(0.0046 * fm**2)
    '0.004599999999999999 fm^2'
    >>> import numpy
    >>> latt = Lattice()
    >>> numpy.array(range(0, 2)) * a(latt)
    array([Value(value=0.0, unit=Unit(mass_dim=-1), scale=Lattice()),
           Value(value=1.0, unit=Unit(mass_dim=-1), scale=Lattice())],
          dtype=object)


To load values in lattice units directly into physical units based on a known
lattice spacing, you can simply multiply the raw numbers by the correct power
of that lattice spacing.

Examples:
    >>> a1 = 0.1 * fm
    >>> str(0.49 / a1)
    '966.9021961200001 MeV'
    >>> import numpy, tempfile
    >>> with tempfile.NamedTemporaryFile() as tmpfile:
    ...     n = tmpfile.write(b'0.49 0.48 0.50')
    ...     tmpfile.flush()
    ...     numpy.loadtxt(tmpfile.name) / a1
    array([Value(value=0.9669021961200001, unit=Unit(mass_dim=1), scale=Physical()),
           Value(value=0.9471694982400001, unit=Unit(mass_dim=1), scale=Physical()),
           Value(value=0.986634894, unit=Unit(mass_dim=1), scale=Physical())],
          dtype=object)


It is also possible to use the Value constructor directly to initialise
values from their mass dimension, but this is not recommended for physical
units as it relies on the internal representation of the units. However, it
works as expected for lattice units.

Example:
    >>> latt = Lattice()
    >>> str(Value(
    ...     16.0,
    ...     Length,
    ...     latt,
    ... ))
    '16.0 a'


Values can be negated, raised to integer powers or rooted.

Examples:
    >>> m2 = 0.0182 * GeV**2
    >>> str(-m2)
    '-0.0182 GeV^2'
    >>> str(m2**3)
    '6.028568000000001e-06 GeV^6'
    >>> str(m2**-1)
    '2.139447063864596 fm^2'
    >>> str(m2.sqrt())
    '134.90737563232042 MeV'
    >>> str((m2**3).root(6))
    '134.90737563232045 MeV'


Values with compatible scales can be added, subtracted, multiplied, divided,
and compared.

Examples:
    >>> m = 0.135 * GeV
    >>> n = 0.135 * GeV
    >>> o = 0.152 * GeV
    >>> str(m + o)
    '287.00000000000006 MeV'
    >>> str(m - o)
    '-16.999999999999986 MeV'
    >>> str(m * o)
    '0.02052 GeV^2'
    >>> str(m / o)
    '0.8881578947368421'
    >>> m == o
    False
    >>> m != o
    True
    >>> m == n
    True
    >>> o > m
    True


In order to perform these operations, values must be converted to the same
scale using v.set_scale(pre_value, post_value), where pre_value is some value
in v's current scale, and post_value is the corresponding value in the target
scale.

Examples:
    >>> latt = Lattice()
    >>> m = 0.135 * GeV
    >>> n = 0.1 / a(latt)
    >>> str(m + n)
    Traceback (most recent call last):
        ...
    ValueError: Can't add values: incompatible scales
    >>> str(m + n.set_scale(a(latt), 0.1 * fm))
    '332.32697880000006 MeV'


Values can be extracted in specific units using the in_unit method.

Example:
    >>> m = 135.0 * MeV
    >>> m.in_unit(fm)
    Traceback (most recent call last):
        ...
    ValueError: Can't convert units: incompatible mass dimensions 1 and -1
    >>> m.in_unit(fm**-1)
    0.6841436524340078
"""

from .scale import Physical, Lattice
from .unit import Mass, Length, Scalar
from .value import Value, Scale

MeV = Value(0.001, Mass, Physical())
GeV = Value(1.0, Mass, Physical())
fm = Value(1.0 / 0.1973269788, Length, Physical())


def a(scale: Scale) -> Value:
    return Value(1.0, Length, scale)
