from ilaff import measurements, units


import numpy as np  # type: ignore

import pytest  # type: ignore


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


def test_cmp() -> None:
    print('blah')


def test_arithmetic() -> None:


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




