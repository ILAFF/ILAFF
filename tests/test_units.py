from ilaff import units
import pytest

def test_mismatched():
    latt = units.Lattice()
    latt2 = units.Lattice()

    a = units.Value(1.0, units.Length, latt)
    a2 = units.Value(1.0, units.Length, latt2)

    with pytest.raises(ValueError):
        two_a = a + a2

    latt.set_scale(a, 0.0904 * units.fm)
    latt2.set_scale(a2, 0.0911 * units.fm)

    with pytest.raises(ValueError):
        two_a = a + a2

    with pytest.raises(ValueError):
        two_a = a / a.in_units(units.Physical())


def test_convert():
    latt = units.Lattice()
    latt2 = units.Lattice()

    a = units.Value(1.0, units.Length, latt)
    a2 = units.Value(1.0, units.Length, latt2)

    latt.set_scale(a, 0.0904 * units.fm)
    latt2.set_scale(a2, 0.0911 * units.fm)

    assert a.in_units(units.Physical()) == 0.0904 * units.fm

    two_a = a.in_units(units.Physical()) + a2.in_units(units.Physical())

    assert two_a == (0.0904 + 0.0911) * units.fm

    m = 0.48 / a
    assert (m.in_units(units.Physical()) / units.GeV).value == pytest.approx(1.04775386973)
