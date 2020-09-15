from ilaff.units.quantity import Quantity
from ilaff.measurements.measurementJack import measurementJack


import numpy as np  # type:ignore

def effE( G: Quantity, deltaT: int, a: Quantity) -> Quantity:
    """
    Effective Mass Function
    """
    EffE: Quantity = ( (1.0/(deltaT*a) ) * np.log( G.value/np.roll(G.value,deltaT) ) )
    return EffE

def effEMJ( GMJ: measurementJack, deltaT: int, a: Quantity) -> measurementJack:
    """
    Effective energy function for measurementJack correlators
    """
    print('TODO: Actually do for jackknifes properly')
    # eh. This seems to work
    return measurementJack(
        GMJ.iValue,
        effE( GMJ.dValue, deltaT, a ),
        effE( GMJ.jackDV, deltaT, a )
    )
