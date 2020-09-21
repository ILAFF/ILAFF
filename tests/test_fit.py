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
    A = 1.0
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

    assert result['A'] == pytest.approx(1.0, 5e-3)
    assert result['E'] == pytest.approx(900 * a / hbarc, 1e-2)




def test_fit_units() -> None:
    latt = units.scale.Lattice()
    w0_phys = 0.17236 * units.fm
    a = 0.07 * units.fm
    A = 1.0
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
            "Gpi": (["configuration", "t"], G * units.one),
        },
        coords = {
            "t": t * units.a(latt),
        },
    )

    databar = fit.jackknife(data)

    result = fit.fit_jack(databar, "Gpi", lambda t, A, E: A * numpy.exp(-t * E), A=1.0, E=0.1 / units.a(latt))

    assert result['A'].data.in_unit(units.one) == pytest.approx(1.0, 5e-3)
    assert result['E'].data.set_scale(units.a(latt), a).in_unit(units.MeV) == pytest.approx(900, 1e-2)
