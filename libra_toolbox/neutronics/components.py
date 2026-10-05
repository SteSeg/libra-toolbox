from dataclasses import dataclass
from typing import Tuple, Optional
import openmc
import materials
import numpy as np
from scipy.optimize import brentq


@dataclass(kw_only=True)
class Component:
    """Base experiment component.

    Position and rotation are expressed in the experiment coordinate system.
    """

    name: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class LeadBrick(Component):
    width: float = 8.0
    length: float = 16.0
    height: float = 4.0
    material: openmc.Material = materials.Lead

    def geometry(self):
        box = openmc.model.RectangularParallelepiped(
            -self.width / 2,
            +self.width / 2,
            -self.length / 2,
            +self.length / 2,
            0,
            self.height,
        )

        cell = openmc.Cell(
            name=self.name,
            region=-box,
            fill=self.material,
        )

        return [cell]

@dataclass
class GeneratorSupport(Component):
    """Support structure for the neutron generator."""

    width: float = 2.54
    length: float = 2.54
    height: float = 2.54

    separation: float = 10.0

    material: openmc.Material = materials.HDPE

    def geometry(self):
        """Build the two generator support blocks."""

        cells = []

        for i, y in enumerate(
            (-self.separation / 2, self.separation / 2),
            start=1,
        ):
            box = openmc.model.RectangularParallelepiped(
                -self.width / 2,
                +self.width / 2,
                y - self.length / 2,
                y + self.length / 2,
                0.0,
                self.height,
            )

            cells.append(
                openmc.Cell(
                    name=f"{self.name}_{i}",
                    region=-box,
                    fill=self.material,
                )
            )

        return cells

@dataclass
class InsulatorLagging(Component):
    radius: float = 12.853
    thickness: float = 0.635
    material: openmc.Material = materials.Alumina

    def geometry(self):
        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.thickness)
        outer = openmc.ZCylinder(r=self.radius)

        region = +bottom & -top & -outer

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class Furnace(Component):
    inner_radius: float = 9.144
    outer_radius: float = 12.002
    thickness: float = 15.24
    material: openmc.Material = materials.Firebrick

    def geometry(self):
        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.thickness)

        inner = openmc.ZCylinder(r=self.inner_radius)
        outer = openmc.ZCylinder(r=self.outer_radius)

        region = (
            +bottom
            & -top
            & +inner
            & -outer
        )

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class InsulatorSlab(Component):
    radius: float = 50.0
    thickness: float = 2.54
    material: openmc.Material = materials.Epoxy

    def geometry(self):
        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.thickness)
        outer = openmc.ZCylinder(r=self.radius)

        region = +bottom & -top & -outer

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class Heater(Component):
    radius: float = 0.439
    height: float = 25.40
    material: openmc.Material = materials.Heater_mat

    def geometry(self):
        cylinder = openmc.ZCylinder(r=self.radius)

        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.height)

        region = +bottom & -top & -cylinder

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class Table(Component):
    width: float = 100.0
    length: float = 100.0
    thickness: float = 2.54
    material: openmc.Material = materials.Epoxy

    def geometry(self):
        x_min = -self.width / 2
        x_max = self.width / 2
        y_min = -self.length / 2
        y_max = self.length / 2

        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.thickness)

        region = (
            +bottom
            & -top
            & +openmc.XPlane(x0=x_min)
            & -openmc.XPlane(x0=x_max)
            & +openmc.YPlane(y0=y_min)
            & -openmc.YPlane(y0=y_max)
        )

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class Vessel1L(Component):
    radius: float = 12.853
    external_radius: float = 13.272

    base_thickness: float = 0.786
    height: float = 21.093
    cover_thickness: float = 2.392

    material: openmc.Material = None

    @property
    def fill_bottom(self):
        return self.base_thickness

    @property
    def fill_top(self):
        return self.base_thickness + self.height

    @property
    def fill_region(self):
        inner = openmc.ZCylinder(r=self.radius)

        bottom = openmc.ZPlane(z0=self.fill_bottom)
        top = openmc.ZPlane(z0=self.fill_top)

        return +bottom & -top & -inner

    def geometry(self):
        outer = openmc.ZCylinder(r=self.external_radius)
        inner = openmc.ZCylinder(r=self.radius)

        bottom = openmc.ZPlane(z0=0.0)
        wall_bottom = openmc.ZPlane(z0=self.fill_bottom)
        wall_top = openmc.ZPlane(z0=self.fill_top)
        top = openmc.ZPlane(
            z0=self.fill_top + self.cover_thickness
        )

        bottom_region = (
            +bottom
            & -wall_bottom
            & -outer
        )

        wall_region = (
            +wall_bottom
            & -wall_top
            & +inner
            & -outer
        )

        top_region = (
            +wall_top
            & -top
            & -outer
        )

        cell = openmc.Cell(
            name=self.name,
            region=(
                bottom_region
                | wall_region
                | top_region
            ),
            fill=self.material,
        )

        return [cell]

    def volume_below(self, z):
        """Available vessel volume below z [cm3]."""

        z = np.clip(
            z,
            self.fill_bottom,
            self.fill_top,
        )

        height = z - self.fill_bottom

        # Simple vessel for now.
        return np.pi * self.radius**2 * height

    def height_for_volume(self, volume):
        """Return the fill height corresponding to a volume [cm3]."""

        if volume <= 0:
            raise ValueError("Volume must be positive.")

        total_volume = self.volume_below(self.fill_top)

        if volume > total_volume:
            raise ValueError(
                f"Requested volume ({volume:.3f} cm3) exceeds "
                f"the available vessel volume ({total_volume:.3f} cm3)."
            )

        z = brentq(
            lambda z: self.volume_below(z) - volume,
            self.fill_bottom,
            self.fill_top,
        )

        return z - self.fill_bottom
    
@dataclass
class Breeder(Component):
    material: openmc.Material

    density: float | None = None
    mass: float | None = None
    volume: float | None = None

    @property
    def requested_volume(self):
        if self.volume is not None:
            return self.volume

        if self.mass is not None:
            if self.density is None:
                raise ValueError(
                    "Density is required when mass is specified."
                )
            return self.mass / self.density

        raise ValueError(
            "Specify either volume or mass."
        )

    def region(self, vessel):
        fill_height = vessel.height_for_volume(
            self.requested_volume
        )

        bottom = openmc.ZPlane(
            z0=vessel.fill_bottom
        )

        top = openmc.ZPlane(
            z0=vessel.fill_bottom + fill_height
        )

        return (
            vessel.fill_region
            & +bottom
            & -top
        )

@dataclass
class HeadSpace:
    name: str
    material: openmc.Material


@dataclass
class OuterVessel(Component):
    """Outer cylindrical vessel wall."""

    inner_radius: float = 12.853
    outer_radius: float = 13.272
    height: float = 21.093

    bottom_thickness: float = 0.786
    top_thickness: float = 2.392

    material: openmc.Material = None

    def geometry(self):
        # Main cylindrical wall
        inner = openmc.ZCylinder(r=self.inner_radius)
        outer = openmc.ZCylinder(r=self.outer_radius)

        bottom = openmc.ZPlane(z0=0.0)
        wall_bottom = openmc.ZPlane(z0=self.bottom_thickness)
        wall_top = openmc.ZPlane(
            z0=self.bottom_thickness + self.height
        )
        top = openmc.ZPlane(
            z0=self.bottom_thickness
            + self.height
            + self.top_thickness
        )

        wall_region = (
            +wall_bottom
            & -wall_top
            & +inner
            & -outer
        )

        bottom_region = (
            +bottom
            & -wall_bottom
            & -outer
        )

        top_region = (
            +wall_top
            & -top
            & -outer
        )

        region = wall_region | bottom_region | top_region

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]

@dataclass
class OuterVesselSweepGas(Component):
    """Sweep gas filling the region inside the outer vessel but
    outside the inner Vessel1L and Furnace.

    Parameters
    ----------
    material : openmc.Material
        Material filling the sweep-gas region.
    """

    material: openmc.Material

    def region(
        self,
        outer_vessel,
        vessel_1l,
        furnace=None,
    ):
        """Return the sweep-gas region."""

        # Start with the entire internal volume of the outer vessel.
        region = outer_vessel.fill_region

        # Exclude the inner Vessel1L.
        region &= ~vessel_1l.region

        # Exclude the Furnace, if present.
        if furnace is not None:
            region &= ~furnace.region

        return region

    def geometry(
        self,
        outer_vessel,
        vessel_1l,
        furnace=None,
    ):
        """Build the OpenMC sweep-gas cell."""

        region = self.region(
            outer_vessel=outer_vessel,
            vessel_1l=vessel_1l,
            furnace=furnace,
        )

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]

from dataclasses import dataclass
import openmc


@dataclass
class BreederVessel1LAssembly:
    name: str
    vessel: Vessel1L
    breeder: Breeder
    headspace: HeadSpace

    def geometry(self):
        # Vessel structure
        vessel_cells = self.vessel.geometry()

        # Breeder region inside the vessel
        breeder_region = self.breeder.region(self.vessel)

        breeder_cell = openmc.Cell(
            name=self.breeder.name,
            region=breeder_region,
            fill=self.breeder.material,
        )

        # Everything inside the vessel not occupied by breeder
        headspace_region = self.vessel.fill_region & ~breeder_region

        headspace_cell = openmc.Cell(
            name=self.headspace.name,
            region=headspace_region,
            fill=self.headspace.material,
        )

        return vessel_cells + [
            breeder_cell,
            headspace_cell,
        ]