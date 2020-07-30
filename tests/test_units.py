from ilaff import units
import pytest  # type: ignore


def test_mismatched() -> None:
    latt = units.Lattice()
    latt2 = units.Lattice()

    a = units.Quantity(1.0, units.Length, latt)
    a2 = units.Quantity(1.0, units.Length, latt2)

    with pytest.raises(ValueError):
        a + a2

    with pytest.raises(ValueError):
        a + a2

    with pytest.raises(ValueError):
        a / a.set_scale(a, 0.0904 * units.fm)


def test_convert() -> None:
    latt = units.Lattice()
    latt2 = units.Lattice()

    a = units.Quantity(1.0, units.Length, latt)
    a2 = units.Quantity(1.0, units.Length, latt2)

    a_phys = a.set_scale(a, 0.0904 * units.fm)
    with pytest.raises(ValueError):
        a2_phys = a2.set_scale(a, 0.0904 * units.fm)
    with pytest.raises(ValueError):
        a2.set_scale(a2**2, 0.0911 * units.fm)
    with pytest.raises(ValueError):
        a2.set_scale(a2, 2 * a2)
    a2_phys = a2.set_scale(a2, 0.0911 * units.fm)
    with pytest.raises(ValueError):
        a2_phys.set_scale(a2, 0.0911 * units.fm)

    assert a_phys == 0.0904 * units.fm

    two_a = a_phys + a2_phys

    assert two_a == (0.0904 + 0.0911) * units.fm

    m = 0.48 / a_phys
    with pytest.raises(ValueError):
        m.in_unit(units.a(latt))
    with pytest.raises(ValueError):
        m.in_unit(units.fm)
    assert m.in_unit(units.GeV) == pytest.approx(1.04775386973)


def test_format() -> None:
    assert str(units.Length) == "length"
    assert str(units.Mass) == "mass"
    assert str(units.Length**4) == "length^4"
    assert str(units.Length**-2) == "mass^2"
    assert str(units.Scalar) == "scalar"

    with pytest.raises(TypeError):
        units.Quantity(
            0.0,
            units.Length,
            units.Scale()  # type: ignore
        )

    latt = units.Lattice()
    t_current = (21 - 16) * units.a(latt)
    m_pi = 0.068 / units.a(latt)
    m_N = 0.49 / units.a(latt)

    assert str(t_current) == '5.0 a'
    assert str(m_N) == '0.49 a^-1'
    assert "{:.4f}".format(1 / m_N**3) == '8.4999 a^3'
    assert "{:.6f}".format(m_pi**2) == '0.004624 a^-2'
    assert "{}".format(m_N * t_current) == '2.45'
    assert str(m_N * t_current) == '2.45'

    t_current = t_current.set_scale(units.a(latt), 0.0913 * units.fm)
    m_pi = m_pi.set_scale(units.a(latt), 0.0913 * units.fm)
    m_N = m_N.set_scale(units.a(latt), 0.0913 * units.fm)

    assert "{:+.7f}".format(t_current / 2.0) == '+0.2282500 fm'
    assert "{:08.5f}".format(m_pi) == '00.14697 GeV'
    assert "{:.1f}".format(m_pi**-2) == '1.8 fm^2'
    assert "{:.5f}".format(m_N * m_pi**2) == '0.02287 GeV^3'
    assert "{:.8f}".format(t_current * m_N) == '2.45000000'


def test_roots() -> None:
    assert units.Dimension(-4)**0.5 == units.Dimension(-2)

    latt = units.Lattice()
    t_current = (21 - 16) * units.a(latt)
    m_pi_squared = 0.004624 / units.a(latt)**2
    m_N = 0.49 / units.a(latt)

    with pytest.raises(ValueError):
        t_current**0.5

    with pytest.raises(ValueError):
        m_pi_squared**(1 / 3)

    with pytest.raises(ValueError):
        m_N**(-1 / 4)

    assert ((
        m_pi_squared
    )**0.5).in_unit(
        1 / units.a(latt)
    ) == pytest.approx(0.068)
    assert ((
        t_current / m_pi_squared / m_N
    )**0.25).in_unit(
        units.a(latt)
    ) == pytest.approx((5 / 0.068 / 0.068 / 0.49)**0.25)
    assert ((
        m_pi_squared * m_N
    )**(-1 / 3)).in_unit(
        units.a(latt)
    ) == pytest.approx((1 / 0.068 / 0.068 / 0.49)**(1 / 3))

    m_N = m_N.set_scale(units.a(latt), 0.0913 * units.fm)

    assert ((
        m_N**2
    )**0.5).in_unit(
        units.MeV
    ) == pytest.approx(0.49 / 0.0913 * 197.3269788)

    assert ((
        m_pi_squared**3 / t_current**2
    )**0.25).set_scale(
        units.a(latt), 0.0913 * units.fm
    ).in_unit(
        units.fm**-2
    ) == pytest.approx((0.068**3 / 5)**0.5 / 0.0913**2)


def test_cmp() -> None:
    latt = units.Lattice()
    t_current = (21 - 16) * units.a(latt)
    t_current_2 = 5 * units.a(latt)
    t_sink = (30 - 16) * units.a(latt)
    m_pi_squared = 0.004624 / units.a(latt)**2
    m_N = 0.49 / units.a(latt)

    assert t_current != t_sink
    assert not (t_current == t_sink)
    assert t_current == t_current_2

    assert t_current >= t_current_2
    assert not (t_current > t_current_2)

    assert t_current <= t_current_2
    assert not (t_current < t_current_2)

    assert t_sink >= t_current
    assert t_sink > t_current
    assert not (t_current >= t_sink)
    assert not (t_current > t_sink)

    assert not (t_sink <= t_current)
    assert not (t_sink < t_current)
    assert t_current <= t_sink
    assert t_current < t_sink

    t_phys = t_current.set_scale(units.a(latt), 0.0913 * units.fm)

    for other in (t_phys, m_pi_squared, m_N, 0.0):
        with pytest.raises(ValueError):
            t_current == other

        with pytest.raises(ValueError):
            t_current != other

        with pytest.raises(ValueError):
            t_current < other

        with pytest.raises(ValueError):
            t_current <= other

        with pytest.raises(ValueError):
            t_current > other

        with pytest.raises(ValueError):
            t_current >= other

    m_phys = m_pi_squared.set_scale(units.a(latt), 0.0898 * units.fm)**0.5
    assert 1 / m_phys != t_phys
    assert not (1 / m_phys == t_phys)
    assert 1 / m_phys > t_phys
    assert m_phys < 1 / t_phys
    assert 1 / m_phys >= t_phys
    assert m_phys <= 1 / t_phys
    assert not (1 / m_phys < t_phys)
    assert not (m_phys > 1 / t_phys)
    assert not (1 / m_phys <= t_phys)
    assert not (m_phys >= 1 / t_phys)

    assert m_phys * t_phys == 0.3456792873051225
    assert 0.3456792873051225 == m_phys * t_phys
    assert not (m_phys * t_phys != 0.3456792873051225)
    assert not (0.3456792873051225 != m_phys * t_phys)
    assert not (m_phys * t_phys > 0.3456792873051225)
    assert not (0.3456792873051225 < m_phys * t_phys)
    assert not (m_phys * t_phys < 0.3456792873051225)
    assert not (0.3456792873051225 > m_phys * t_phys)
    assert m_phys * t_phys >= 0.3456792873051225
    assert 0.3456792873051225 <= m_phys * t_phys
    assert m_phys * t_phys <= 0.3456792873051225
    assert 0.3456792873051225 >= m_phys * t_phys
    assert not (m_phys * t_phys == 0.3)
    assert not (0.3 == m_phys * t_phys)
    assert m_phys * t_phys != 0.3
    assert 0.3 != m_phys * t_phys
    assert m_phys * t_phys > 0.3
    assert 0.3 < m_phys * t_phys
    assert not (m_phys * t_phys < 0.3)
    assert not (0.3 > m_phys * t_phys)
    assert m_phys * t_phys >= 0.3
    assert 0.3 <= m_phys * t_phys
    assert not (m_phys * t_phys <= 0.3)
    assert not (0.3 >= m_phys * t_phys)
    assert not (m_phys * t_phys > 0.4)
    assert not (0.4 < m_phys * t_phys)
    assert m_phys * t_phys < 0.4
    assert 0.4 > m_phys * t_phys
    assert not (m_phys * t_phys >= 0.4)
    assert not (0.4 <= m_phys * t_phys)
    assert m_phys * t_phys <= 0.4
    assert 0.4 >= m_phys * t_phys


def test_arithmetic() -> None:
    latt = units.Lattice()
    t_current = (21 - 16) * units.a(latt)
    t_current_2 = 5 * units.a(latt)
    t_sink = (30 - 16) * units.a(latt)
    m_N = 0.49 / units.a(latt)

    assert -t_current == (-5) * units.a(latt)
    assert (
        t_current + (-t_current_2)
    ).in_unit(
        units.a(latt)
    ) == pytest.approx(0)
    assert t_sink - t_current == 9 * units.a(latt)

    with pytest.raises(ValueError):
        t_sink + m_N

    with pytest.raises(ValueError):
        t_sink - m_N

    t_phys = t_current.set_scale(units.a(latt), 0.0913 * units.fm)

    with pytest.raises(ValueError):
        t_sink + t_phys

    with pytest.raises(ValueError):
        t_sink - t_phys

    with pytest.raises(ValueError):
        t_sink * t_phys

    with pytest.raises(ValueError):
        t_sink / t_phys

    with pytest.raises(ValueError):
        t_sink + 1.0

    with pytest.raises(ValueError):
        t_sink - 1.0

    with pytest.raises(ValueError):
        1.0 + t_sink

    with pytest.raises(ValueError):
        1.0 - t_sink

    assert t_sink * m_N + 1.0 == 7.859999999999999
    assert t_sink * m_N - 1.0 == 5.859999999999999
    assert 0.5 + t_sink * m_N == 7.359999999999999
    assert 0.5 - t_sink * m_N == -6.359999999999999

    t_phys = t_current.set_scale(units.a(latt), 0.0913 * units.fm)

    assert (t_sink * 2).in_unit(units.a(latt)) == pytest.approx(28)
    assert (0.1 * t_sink).in_unit(units.a(latt)) == pytest.approx(1.4)
    assert (
        t_sink * (
            t_sink.set_scale(units.a(latt), 0.0913 * units.fm)
            * m_N.set_scale(units.a(latt), 0.0913 * units.fm)
        )
    ).in_unit(
        units.a(latt)
    ) == pytest.approx(14 * 6.86)
    assert (
        (t_sink * m_N) * t_phys
    ).in_unit(
        units.fm
    ) == pytest.approx(5 * 0.0913 * 6.86)

    assert (
        t_sink / 2
    ).in_unit(
        units.a(latt)
    ) == pytest.approx(7)
    assert (
        0.1 / t_sink
    ).in_unit(
        units.a(latt)**-1
    ) == pytest.approx(0.00714285714)
    assert (
        t_sink / (
            t_sink.set_scale(units.a(latt), 0.0913 * units.fm)
            * m_N.set_scale(units.a(latt), 0.0913 * units.fm)
        )
    ).in_unit(
        units.a(latt)
    ) == pytest.approx(14 / 6.86)
    assert (
        (t_sink * m_N) / t_phys
    ).in_unit(
        units.fm**-1
    ) == pytest.approx(6.86 / (5 * 0.0913))


def test_numpy() -> None:
    import numpy  # type: ignore

    t = numpy.arange(1, 9) * units.fm
    m_eff = numpy.array([1.2, 1.1, 0.9, 0.9, 0.8, 0.9, 0.9, 0.8]) * units.GeV

    assert len(m_eff) == len(t)
    assert len(t) == 8

    assert m_eff[1] == 1.1 * units.GeV
    assert m_eff[5] == 0.9 * units.GeV

    assert numpy.all(m_eff[2:5] == numpy.array([0.9, 0.9, 0.8]) * units.GeV)

    assert numpy.all(t[m_eff < 0.85 * units.GeV] == numpy.array([5.0, 8.0]) * units.fm)

    it = iter(t)
    assert next(it) == 1.0 * units.fm
    assert next(it) == 2.0 * units.fm
    assert next(it) == 3.0 * units.fm
    assert next(it) == 4.0 * units.fm
    assert next(it) == 5.0 * units.fm
    assert next(it) == 6.0 * units.fm
    assert next(it) == 7.0 * units.fm
    assert next(it) == 8.0 * units.fm
    with pytest.raises(StopIteration):
        next(it)

    it = reversed(m_eff)
    assert next(it) == 0.8 * units.GeV
    assert next(it) == 0.9 * units.GeV
    assert next(it) == 0.9 * units.GeV
    assert next(it) == 0.8 * units.GeV
    assert next(it) == 0.9 * units.GeV
    assert next(it) == 0.9 * units.GeV
    assert next(it) == 1.1 * units.GeV
    assert next(it) == 1.2 * units.GeV
    with pytest.raises(StopIteration):
        next(it)
