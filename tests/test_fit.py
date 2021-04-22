from ilaff import fit, units
from xarray import Dataset, Variable
import numpy
import pytest


def test_jack() -> None:
    w0_phys = 0.17236
    i = numpy.array([0, 1, 2, 3])
    a = numpy.array([0.07, 0.072, 0.069, 0.071])
    A = numpy.array([1.1, 1.2, 1.0, 0.8])
    E = numpy.array([0.05, 0.046, 0.048, 0.049])
    t = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    data = Dataset(
        {
            "w0": (["configuration"], w0_phys / a),
            "Gpi": (["configuration", "t"], A.reshape(*A.shape, 1) * numpy.exp(-numpy.outer(E, t))),
        },
        coords={
            "t": t,
        }
    )

    databar = fit.jackknife(data)

    assert databar["w0"].isel(jack=0) == numpy.mean(w0_phys / a)
    for n in range(4):
        assert databar["w0"].isel(jack=n + 1) == numpy.mean(w0_phys / a[i != n])


def test_jack_units() -> None:
    latt = units.scale.Lattice()
    w0_phys = 0.17236 * units.fm
    i = numpy.array([0, 1, 2, 3])
    a = numpy.array([0.07, 0.072, 0.069, 0.071]) * units.fm
    A = numpy.array([1.1, 1.2, 1.0, 0.8])
    E = numpy.array([0.05, 0.046, 0.048, 0.049]) / units.a(latt)
    t = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    data = Dataset(
        {
            "w0": (["configuration"], w0_phys / a * units.a(latt)),
            "Gpi": (["configuration", "t"], A.reshape(*A.shape, 1) * numpy.exp(-numpy.outer(E, t) * units.a(latt))),
        },
        coords={
            "t": t,
        }
    )

    databar = fit.jackknife(data)

    assert databar["w0"].isel(jack=0) == numpy.mean(w0_phys / a * units.a(latt))
    for n in range(4):
        assert databar["w0"].isel(jack=n + 1) == numpy.mean(w0_phys / a[i != n] * units.a(latt))


def test_fit() -> None:
    w0_phys = 0.17236
    a = 0.07
    A = 1.2
    E = 900
    hbarc = 197.3
    t = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    sigma = 0.1 * numpy.exp(-3*135/2*t*a/hbarc)

    rng = numpy.random.default_rng(1)

    G = A * numpy.exp(-E*t*a/hbarc) + rng.normal(size=(200, len(t))) * sigma

    data = Dataset(
        {
            "w0": (["configuration"], w0_phys / a + rng.normal(size=(200,)) * 0.1),
            "Gpi": (["configuration", "t"], G),
        },
        coords={
            "t": t,
        }
    )

    databar = fit.jackknife(data)

    result = fit.fit_jack(databar, "Gpi", lambda t, A, E: A * numpy.exp(-t * E), A=1.0, E=0.1)

    assert result['A'] == pytest.approx(A, 5e-3)
    assert result['E'] == pytest.approx(E * a / hbarc, 1e-2)




def test_fit_units() -> None:
    latt = units.scale.Lattice()
    w0_phys = 0.17236 * units.fm
    a = 0.07 * units.fm
    A = 1.2
    E = 900 * units.MeV
    mpi = 135 * units.MeV
    t = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    sigma = 0.1 * numpy.exp(-3/2 * mpi * t * a)

    rng = numpy.random.default_rng(1)

    G = A * numpy.exp(-E * t * a) + rng.normal(size=(200, len(t))) * sigma

    # TODO: figure out why coords do not work
    data = Dataset(
        {
            "w0": (["configuration"], w0_phys.set_scale(a, units.a(latt)) + rng.normal(size=(200,)) * 0.1 * units.a(latt)),
            # "Gpi": (["configuration", "t"], G * units.one),
            "Gpi": (["configuration", "time"], G * units.one),
            "t": (["time"], t * units.a(latt)),
        },
        coords = {
            # "t": t * units.a(latt),
        },
    )

    databar = fit.jackknife(data)

    result = fit.fit_jack(databar, "Gpi", lambda t, A, E: A * numpy.exp(-t * E), A=1.0, E=0.1 / units.a(latt))

    assert result['A'].data.in_unit(units.one) == pytest.approx(A, 5e-3)
    assert result['E'].data.set_scale(units.a(latt), a).in_unit(units.MeV) == pytest.approx(E.in_unit(units.MeV), 1e-2)


def test_model() -> None:
    exp = fit.Model.new(lambda A, x: A * numpy.exp(x))
    sin = fit.Model.new(lambda x: numpy.sin(x))

    t0 = 0.2345
    t = numpy.linspace(0.1, 10.4)

    A0 = 127
    A = numpy.linspace(-28, 46)

    for mul in (sin * 2, 2 * sin):
        assert mul(t0) == 2.0 * numpy.sin(t0)
        assert numpy.all(mul(t) == 2.0 * numpy.sin(t))

    for mul in (sin * units.MeV, units.MeV * sin):
        assert mul(t0) == numpy.sin(t0) * units.MeV
        assert numpy.all(mul(t) == numpy.sin(t) * units.MeV)

    neg = -exp
    assert neg(A0, t0) == -A0 * numpy.exp(t0)
    assert numpy.all(neg(A, t) == -A * numpy.exp(t))

    for add in (sin + exp, exp + sin, (lambda x: numpy.sin(x)) + exp, exp + (lambda x: numpy.sin(x))):
        assert add(A=A0, x=t0) == numpy.sin(t0) + A0 * numpy.exp(t0)
        assert numpy.all(add(A=A0, x=t) == numpy.sin(t) + A0 * numpy.exp(t))
        assert numpy.all(add(A=A, x=t0) == numpy.sin(t0) + A * numpy.exp(t0))
        assert numpy.all(add(A=A, x=t) == numpy.sin(t) + A * numpy.exp(t))

    pos = +exp
    assert numpy.all(pos(A, t) == exp(A, t))

    for sub in (sin - exp, -(exp - sin), (lambda x: numpy.sin(x)) - exp, sin - (lambda A, x: A * numpy.exp(x))):
        assert numpy.all(sub(A=A, x=t) == numpy.sin(t) - A * numpy.exp(t))
