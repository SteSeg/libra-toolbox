from collections.abc import Iterable
from libra_toolbox.neutronics.neutron_source import A325_generator_diamond

import numpy as np

import pytest


def test_get_avg_neutron_rate():
    try:
        import openmc
    except ImportError:
        pytest.skip("OpenMC is not installed")

    source = A325_generator_diamond((0, 0, 0), (0, 0, 1))

    assert isinstance(source, Iterable)
    for s in source:
        assert isinstance(s, openmc.IndependentSource)


@pytest.mark.parametrize(
    "uvw",
    [
        (0, 0, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, -1),
        (-1, 0, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 0, 2),
    ],
)
def test_reference_direction(uvw):
    """Any beam axis must work, including (1, 0, 0). OpenMC >= 0.15.3 raises a
    ValueError for it unless reference_vwu is given."""
    pytest.importorskip("openmc")

    sources = A325_generator_diamond((0, 0, 0), uvw)

    expected = np.array(uvw, dtype=float)
    expected /= np.linalg.norm(expected)
    assert len(sources) > 0
    for s in sources:
        np.testing.assert_allclose(s.angle.reference_uvw, expected)
        vwu = getattr(s.angle, "reference_vwu", None)
        if vwu is not None:  # OpenMC >= 0.15.3
            assert abs(np.dot(vwu, s.angle.reference_uvw)) < 1e-12
            assert np.isclose(np.linalg.norm(vwu), 1.0)


@pytest.mark.parametrize("uvw", [(0, 0, 1), (1, 1, 1), (1, 0, 1), (3, 4, 0)])
def test_unbroken_axes_keep_openmc_default(uvw):
    """Axes OpenMC already accepted must get exactly the angle distribution
    OpenMC builds by itself, so existing models reproduce particle by particle."""
    openmc = pytest.importorskip("openmc")

    for s in A325_generator_diamond((0, 0, 0), uvw):
        reference = openmc.stats.PolarAzimuthal(
            mu=s.angle.mu, phi=s.angle.phi, reference_uvw=uvw
        )
        np.testing.assert_array_equal(s.angle.reference_uvw, reference.reference_uvw)
        if hasattr(reference, "reference_vwu"):  # OpenMC >= 0.15.3
            np.testing.assert_array_equal(
                s.angle.reference_vwu, reference.reference_vwu
            )


@pytest.mark.parametrize("uvw", [(0, 0, 0), (np.nan, 0, 1)])
def test_invalid_reference_direction_raises(uvw):
    pytest.importorskip("openmc")
    with pytest.raises(ValueError):
        A325_generator_diamond((0, 0, 0), uvw)


def test_sources_export_to_xml(tmp_path):
    """The sources written to settings.xml carry the beam axis."""
    openmc = pytest.importorskip("openmc")

    settings = openmc.Settings()
    settings.source = A325_generator_diamond((1, 2, 3), (1, 0, 0))
    settings.export_to_xml(tmp_path / "settings.xml")
    text = (tmp_path / "settings.xml").read_text()
    assert text.count('reference_uvw="1.0 0.0 0.0"') == len(settings.source)


def _sample_directions(sources, n, workdir):
    """Samples n source sites with OpenMC's own sampler (void model, no
    cross sections needed) and returns the unit direction of each."""
    import os

    import openmc
    import openmc.lib

    sph = openmc.Sphere(r=1e4, boundary_type="vacuum")
    model = openmc.Model(geometry=openmc.Geometry([openmc.Cell(region=-sph)]))
    model.settings.run_mode = "fixed source"
    model.settings.particles = 100
    model.settings.batches = 1
    model.settings.source = sources
    # a void model still needs a cross_sections file that lists one library,
    # but the library itself is never read
    xs = os.path.join(workdir, "cross_sections.xml")
    open(os.path.join(workdir, "dummy.h5"), "w").close()
    with open(xs, "w") as f:
        f.write(
            '<?xml version="1.0"?>\n<cross_sections>\n'
            '<library materials="H1" path="dummy.h5" type="neutron"/>\n'
            "</cross_sections>\n"
        )
    old_xs = os.environ.get("OPENMC_CROSS_SECTIONS")
    os.environ["OPENMC_CROSS_SECTIONS"] = xs
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        if hasattr(model, "export_to_model_xml"):
            model.export_to_model_xml()
        else:
            model.export_to_xml()
        openmc.lib.init()
        try:
            sites = openmc.lib.sample_external_source(n, prn_seed=12345)
        finally:
            openmc.lib.finalize()
    finally:
        os.chdir(cwd)
        if old_xs is None:
            del os.environ["OPENMC_CROSS_SECTIONS"]
        else:
            os.environ["OPENMC_CROSS_SECTIONS"] = old_xs
    return np.array([s.u for s in sites])


@pytest.mark.parametrize("uvw", [(1, 0, 0), (0, 1, 1)])
def test_sampled_directions_follow_tabulated_polar_distribution(uvw, tmp_path):
    """Sample the source and check the physics, not just that it builds:
    directions are unit length, the cosine of the polar angle about the beam
    axis follows the tabulated (strength-weighted piecewise uniform) mu
    distribution, and the azimuth about the axis is uniform."""
    pytest.importorskip("openmc.lib")
    n = 20000
    sources = A325_generator_diamond((0, 0, 0), uvw)
    u = _sample_directions(sources, n, tmp_path)

    # unit length
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-9)

    axis = np.array(uvw, dtype=float)
    axis /= np.linalg.norm(axis)
    mu = u @ axis

    # expected CDF of mu: mixture of uniform bins weighted by source strength
    a = np.array([s.angle.mu.a for s in sources])
    b = np.array([s.angle.mu.b for s in sources])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    w = np.array([s.strength for s in sources])
    w = w / w.sum()

    ms = np.sort(mu)
    cdf = np.clip((ms[:, None] - lo) / (hi - lo), 0.0, 1.0) @ w
    ks = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(0, n) / n))
    assert ks < 1.95 / np.sqrt(n), f"polar KS statistic {ks:.4f}"  # p ~ 1e-3

    # azimuth about the axis is uniform (Rayleigh test, p = exp(-16) at bound)
    e1 = np.cross(axis, [0.3, 0.5, 0.8])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    perp = u - mu[:, None] * axis
    phi = np.arctan2(perp @ e2, perp @ e1)
    assert np.abs(np.mean(np.exp(1j * phi))) < 4.0 / np.sqrt(n)
