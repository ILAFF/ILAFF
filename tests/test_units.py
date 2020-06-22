from ilaff import units
import pytest

def test_mismatched():
    latt = units.Lattice()
    latt2 = units.Lattice()

    a = units.Value(1.0, units.Length, latt)
    a2 = units.Value(1.0, units.Length, latt2)

    with pytest.raises(ValueError):
        two_a = a + a2

    with pytest.raises(ValueError):
        two_a = a + a2

    with pytest.raises(ValueError):
        two_a = a / a.set_scale(a, 0.0904 * units.fm)


def test_convert():
    latt = units.Lattice()
    latt2 = units.Lattice()

    a = units.Value(1.0, units.Length, latt)
    a2 = units.Value(1.0, units.Length, latt2)

    a_phys = a.set_scale(a, 0.0904 * units.fm)
    with pytest.raises(ValueError):
        a2_phys = a2.set_scale(a, 0.0904 * units.fm)
    a2_phys = a2.set_scale(a2, 0.0911 * units.fm)

    assert a_phys == 0.0904 * units.fm

    two_a = a_phys + a2_phys

    assert two_a == (0.0904 + 0.0911) * units.fm

    m = 0.48 / a_phys
    assert m.in_unit(units.GeV) == pytest.approx(1.04775386973)
