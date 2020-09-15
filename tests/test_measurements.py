from ilaff import measurements, units


import numpy as np  # type: ignore

import pytest # type: ignore

#TEST MEASUREMENTJACK

def test_Jackmismatched() -> None:
    """
    Testing that adding measurements of different lattice's doesn't work
    """
    latt1 = units.Lattice()
    latt2 = units.Lattice()

    a1 = units.Quantity(1.0, units.Length, latt1)
    a2 = units.Quantity(1.0, units.Length, latt2)

    t1 = np.arange(1,64)
    G1 = np.exp( -0.5*t1) * a1
    jackG1 = np.random.random_sample( [64,200] ) * a1
    GMeas1 = measurements.measurementJack( t1 * a1, G1, jackG1 )
    t2 = np.arange(1,64)
    G2 = np.exp( -0.5*t2) * a2
    jackG2 = np.random.random_sample( [64,200] ) * a2
    GMeas2 = measurements.measurementJack( t2 * a2, G2, jackG2 )


    with pytest.raises(ValueError):
        GMeas1 + GMeas2

    a1 = a1.set_scale(a1, 0.0904 * units.fm)
    with pytest.raises(ValueError):
        GMeas1 = GMeas1 / a1


def test_Jackarithmetic() -> None:


    latt1 = units.Lattice()
    a1 = units.Quantity(1.0, units.Length, latt1)
    t1 = np.array( [ 1 ] )
    G1 = np.array( -0.5*t1) * a1
    jackG1 = np.array( [ [1,2] ] ) * a1
    GMeas1 = measurements.measurementJack( t1 * a1, G1, jackG1 )
    
    with pytest.raises(ValueError):
        GMeas1 + 2
    with pytest.raises(ValueError):
        GMeas1 - 2
    with pytest.raises(ValueError):
        2 + GMeas1
    with pytest.raises(ValueError):
        2 - GMeas1

    #Checking that adding sums adds only dValue and jackDV
    assert( (GMeas1 + GMeas1).iValue == t1 * a1 )
    assert( (GMeas1 + GMeas1).dValue == 2*G1 )
    assert( ( (GMeas1 + GMeas1).jackDV == 2*jackG1 ).all() )
    #Subtraction
    assert( (GMeas1 - GMeas1).iValue == t1 * a1 )
    assert( (GMeas1 - GMeas1).dValue == 0*G1 )
    assert( ( (GMeas1 - GMeas1).jackDV == 0*jackG1 ).all() )
    #Adding a measurement to a quantity
    with pytest.raises(ValueError):
        G1 + GMeas1

    latt4 = units.Lattice()
    a4 = units.Quantity(1.0, units.Length, latt4)
    G4 = np.array( -0.5*t1) * a4
    with pytest.raises(ValueError):
        GMeas1 + G4
    with pytest.raises(ValueError):
        G4 +GMeas1
    with pytest.raises(ValueError):
        GMeas1 - G4
    with pytest.raises(ValueError):
        G4 - GMeas1

    t2 = np.array( [ 2 ] )        
    G2 = np.array( -0.5*t2) * a1
    GMeas2 = measurements.measurementJack( t2 * a1, G2, jackG1 )

    with pytest.raises(ValueError):
        GMeas1 + GMeas2

    t3 = np.array( [ 1 ] )        
    G3 = np.array( -0.25*t3) * a1
    jackG1 = np.array( [ [0.5,1] ] ) * a1
    GMeas3 = measurements.measurementJack( t3 * a1, G3, jackG1 )

    #Addition
    assert( GMeas3 + GMeas1 == GMeas1 + GMeas3 )
    #Subtraction
    assert( (GMeas3 - GMeas1).iValue == t1 * a1 )
    assert( (GMeas3 - GMeas1).dValue == -0.5 * G1 )
    assert( ( (GMeas3 - GMeas1).jackDV == -1.0 * jackG1 ).all() )
    #
    assert( (GMeas1 - GMeas3).iValue == t1 * a1 )
    assert( (GMeas1 - GMeas3).dValue == 0.5 * G1 )
    assert( ( (GMeas1 - GMeas3).jackDV == 1.0 * jackG1 ).all() )

    assert( GMeas1 * 2 == GMeas1 + GMeas1 )
    assert( 2 * GMeas1 == GMeas1 + GMeas1 )
    assert( -GMeas1 == -1*GMeas1 )

    assert( GMeas1 / 2 == 0.5 * GMeas1)
    #assert( 2/GMeas1 ==  )

    assert( GMeas1**2.0 == GMeas1 * GMeas1 )


#FOR MEASUREMENT

def test_mismatched() -> None:
    """
    Testing that adding measurements of different lattice's doesn't work
    """
    latt1 = units.Lattice()
    latt2 = units.Lattice()

    a1 = units.Quantity(1.0, units.Length, latt1)
    a2 = units.Quantity(1.0, units.Length, latt2)

    t1 = np.arange(1,64)
    G1 = np.exp( -0.5*t1) * a1
    GMeas1 = measurements.measurement( t1 * a1, G1 )
    t2 = np.arange(1,64)
    G2 = np.exp( -0.5*t2) * a2
    GMeas2 = measurements.measurement( t2 * a2, G2)


    with pytest.raises(ValueError):
        GMeas1 + GMeas2

    a1 = a1.set_scale(a1, 0.0904 * units.fm)
    with pytest.raises(ValueError):
        GMeas1 = GMeas1 / a1



def test_arithmetic() -> None:


    latt1 = units.Lattice()
    a1 = units.Quantity(1.0, units.Length, latt1)
    t1 = np.array( [ 1 ] )
    G1 = np.array( -0.5*t1) * a1
    GMeas1 = measurements.measurement( t1 * a1, G1 )
    
    with pytest.raises(ValueError):
        GMeas1 + 2
    with pytest.raises(ValueError):
        GMeas1 - 2
    with pytest.raises(ValueError):
        2 + GMeas1
    with pytest.raises(ValueError):
        2 - GMeas1

    #Checking that adding sums adds only dValue and jackDV
    assert( (GMeas1 + GMeas1).iValue == t1 * a1 )
    assert( (GMeas1 + GMeas1).dValue == 2*G1 )
    #Subtraction
    assert( (GMeas1 - GMeas1).iValue == t1 * a1 )
    assert( (GMeas1 - GMeas1).dValue == 0*G1 )
    #Adding a measurement to a quantity
    with pytest.raises(ValueError):
        G1 + GMeas1

    latt4 = units.Lattice()
    a4 = units.Quantity(1.0, units.Length, latt4)
    G4 = np.array( -0.5*t1) * a4
    with pytest.raises(ValueError):
        GMeas1 + G4
    with pytest.raises(ValueError):
        G4 +GMeas1
    with pytest.raises(ValueError):
        GMeas1 - G4
    with pytest.raises(ValueError):
        G4 - GMeas1

    t2 = np.array( [ 2 ] )        
    G2 = np.array( -0.5*t2) * a1
    GMeas2 = measurements.measurement( t2 * a1, G2)

    with pytest.raises(ValueError):
        GMeas1 + GMeas2

    t3 = np.array( [ 1 ] )        
    G3 = np.array( -0.25*t3) * a1
    GMeas3 = measurements.measurement( t3 * a1, G3 )

    #Addition
    assert( GMeas3 + GMeas1 == GMeas1 + GMeas3 )
    #Subtraction
    assert( (GMeas3 - GMeas1).iValue == t1 * a1 )
    assert( (GMeas3 - GMeas1).dValue == -0.5 * G1 )
    #
    assert( (GMeas1 - GMeas3).iValue == t1 * a1 )
    assert( (GMeas1 - GMeas3).dValue == 0.5 * G1 )

    assert( GMeas1 * 2 == GMeas1 + GMeas1 )
    assert( 2 * GMeas1 == GMeas1 + GMeas1 )
    assert( -GMeas1 == -1*GMeas1 )

    assert( GMeas1 / 2 == 0.5 * GMeas1)
    #assert( 2/GMeas1 ==  )

    assert( GMeas1**2.0 == GMeas1 * GMeas1 )


#mixed?

def test_makeJack() -> None:
    """
    Testing making a jackknife from a set of measurements
    """

    latt1 = units.Lattice()
    a1 = units.Quantity(1.0, units.Length, latt1)
    
    x = np.array( [ 1, 2 ] ) * a1
    y = np.array( [ [ 1, 2 ], [ 3, 5 ], [ 4, 6 ] ] ) / a1
    Meas = measurements.measurement( x, y )
    measJack = Meas.measResample('jack')
    assert( y[0][1] == 2 /a1 )
    assert( ( measJack.iValue == x).all() )
    assert( measJack.dValue[0] == ( (1.0+3.0+4.0)/3.0 ) /a1 )
    assert( measJack.dValue[1] ==  ( (2.0+5.0+6.0)/3.0 ) /a1 )
    assert( ( measJack.jackDV[0] == np.array([3.5,5.5]) /a1 ).all() )
    assert( ( measJack.jackDV[1] == np.array([2.5,4.0]) /a1 ).all() )
    assert( ( measJack.jackDV[2] == np.array([2,3.5]) /a1 ).all() )

    jackErr=measJack.jackerr()
    assert( ( jackErr == np.array( [0.8819171036881968,1.2018504251546631] ) /a1).all() )
