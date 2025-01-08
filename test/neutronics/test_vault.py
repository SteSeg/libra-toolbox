from libra_toolbox.neutronics import vault

import pytest


def test_vault_model():
    """Test that the vault model can be created and run without errors"""
    try:
        import openmc
    except ImportError:
        pytest.skip("OpenMC is not installed")
    import openmc

    point = openmc.stats.Point((500, 200, 100))
    src = openmc.IndependentSource(space=point)
    src.energy = openmc.stats.Discrete([14.1e6], [1.0])

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = src
    settings.batches = 3
    settings.inactive = 0
    settings.particles = int(100)

    water = openmc.Material(name="water")
    water.add_element("O", 1 / 3)
    water.add_element("H", 2 / 3)
    water.set_density("g/cc", 1.0)

    water_sphere = openmc.Sphere(
        r=10, x0=src.space.xyz[0] + 100, y0=src.space.xyz[1], z0=src.space.xyz[2]
    )
    water_cell = openmc.Cell(region=-water_sphere, fill=water)
    overall_exclusion_region = -water_sphere

    model = vault.build_vault_model(
        settings=settings,
        tallies=openmc.Tallies(),
        added_cells=[water_cell],
        added_materials=[water],
        overall_exclusion_region=overall_exclusion_region,
    )

    model.run(geometry_debug=True)
