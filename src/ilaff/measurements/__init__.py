"""Measurement classes for measurements using units

A separate instance of ``Lattice`` should be constructed for each independent
lattice spacing. This ensures that the independent lattice spacings are kept
distinct.

Examples:
    Here we consider i.e, an independent variable t, dependent variable G with 200 jackknife subensembles jackG
    >>> from ilaff import units, measurements as meas
    >>> import numpy as np
    >>> latt = units.Lattice()
    >>> a = units.a(latt)
    >>> t = np.arange(1,64) * units.a(latt)
    >>> G = np.exp( -0.5*t) * units.a(latt)
    >>> jackG = np.random.random_sample( [64,200] ) * units.a(latt)
    >>> 
    >>> GMeas = meas.measurementJack( t, G, jackG )

measurementJack's  with compatible scales and dimensions can be added, subtracted,
multiplied, divided, and compared. They must also have equal independent variables.

Examples:
    >>> GMeas2 = meas.measurementJack( t, G, jackG )
    >>> GMeas2 == GMeas
    True
    >>> GMeas2**2.0 == GMeas2 * GMeas2
    True
    >>> GMeas2 > GMeas
    False

The jackknife error of the dependent variable can be obtained as a Quantity

Examples:
    >>> GjackErr = GMeas.jackerr()


"""




from .measurementJack import measurementJack
