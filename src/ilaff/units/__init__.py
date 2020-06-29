"""Dimensional analysis for use in lattice QCD analyses.

A separate instance of ``Lattice`` should be constructed for each independent
lattice spacing. This ensures that the independent lattice spacings are kept
distinct.

New quantities should generally be constructed by multiplying a float, numpy
array, or other raw value by one of the provided units (i.e. ``units.MeV``, 
``units.GeV``, ``units.fm``, or ``units.a(scale)``).

Examples:
    >>> str(135.0 * MeV)
    '0.135 GeV'
    >>> str(0.0046 * fm**2)
    '0.0046 fm^2'
    >>> import numpy
    >>> latt = Lattice()
    >>> str(0.068 / a(latt))
    '0.068 a^-1'
    >>> numpy.array(range(0, 2)) * a(latt)
    Quantity(value=array([0., 1.]), dimension=Dimension(mass_dim=-1), scale=Lattice())


To load values in lattice units directly into physical units based on a known
lattice spacing, you can simply multiply the raw numbers by the correct power
of that lattice spacing.

Examples:
    >>> a1 = 0.1 * fm
    >>> str(0.49 / a1)
    '0.9669021961200001 GeV'
    >>> import numpy, tempfile
    >>> with tempfile.NamedTemporaryFile() as tmpfile:
    ...     n = tmpfile.write(b'0.49 0.48 0.50')
    ...     tmpfile.flush()
    ...     numpy.loadtxt(tmpfile.name) / a1
    Quantity(value=array([0.9669022 , 0.9471695 , 0.98663489]), dimension=Dimension(mass_dim=1), scale=Physical())


It is also possible to use the ``Quantity`` constructor directly to initialise
quantities from their mass dimension, but this is not recommended for physical
units as it relies on the internal representation of the units in ``Quantity``.
However, it works as expected for lattice units.

Example:
    >>> latt = Lattice()
    >>> str(Quantity(
    ...     16.0,
    ...     Length,
    ...     latt,
    ... ))
    '16.0 a'


Quantities can be negated, raised to integer powers or rooted.

Examples:
    >>> m2 = 0.0182 * GeV**2
    >>> str(-m2)
    '-0.0182 GeV^2'
    >>> str(m2**3)
    '6.028568000000001e-06 GeV^6'
    >>> str(m2**-1)
    '2.1394470638645964 fm^2'
    >>> str(m2.sqrt())
    '0.13490737563232041 GeV'
    >>> str((m2**3).root(6))
    '0.13490737563232044 GeV'


Quantities with compatible scales and dimensions can be added, subtracted,
multiplied, divided, and compared.

Examples:
    >>> m = 0.135 * GeV
    >>> n = 0.135 * GeV
    >>> o = 0.152 * GeV
    >>> str(m + o)
    '0.28700000000000003 GeV'
    >>> str(m - o)
    '-0.016999999999999987 GeV'
    >>> str(m * o + n**2)
    '0.038745 GeV^2'
    >>> str(m / o + 2.0)
    '2.888157894736842'
    >>> str(1.0 / n)
    '1.4616813244444444 fm'
    >>> m == o
    False
    >>> m != o
    True
    >>> m**2 == n * n
    True
    >>> o > m
    True


In order to perform these operations, quantities must be converted to the same
scale using ``q.set_scale(current, target)``, where ``current`` is some
quantity in ``q``'s current scale, and ``target`` is the corresponding quantity
in the target scale.

Examples:
    >>> latt = Lattice()
    >>> m = 0.135 * GeV
    >>> n = 0.1 / a(latt)
    >>> str(m + n)
    Traceback (most recent call last):
        ...
    ValueError: Can't add values: incompatible scales
    >>> str(m + n.set_scale(a(latt), 0.1 * fm))
    '0.33232697880000006 GeV'


Values can be extracted in specific units using the ``in_unit`` method.

Example:
    >>> m = 135.0 * MeV
    >>> m.in_unit(fm)
    Traceback (most recent call last):
        ...
    ValueError: Can't convert units: incompatible mass dimensions 1 and -1
    >>> m.in_unit(fm**-1)
    0.6841436524340078
"""

from .scale import Physical, Lattice, MeV, GeV, fm, a, one
from .dimension import Dimension, Mass, Length, Scalar
from .quantity import Quantity, Scale

__all__ = [
    'Physical', 'Lattice', 'MeV', 'GeV', 'fm', 'a', 'one',
    'Dimension', 'Mass', 'Length', 'Scalar',
    'Quantity', 'Scale',
]
