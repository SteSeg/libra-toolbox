from dataclasses import dataclass
from typing import Tuple, Optional
import openmc
import materials


@dataclass
class Component:
    """Base experiment component.

    Position and rotation are expressed in the experiment coordinate system.
    """

    name: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: Optional[openmc.Material] = None


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

    material: openmc.Material = None

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
class Vessel1L(Component):
    radius: float = 12.853
    external_radius: float = 13.272

    base_thickness: float = 0.786
    height: float = 21.093
    cover_thickness: float = 2.392
    material: openmc.Material = materials.Inconel625

    def geometry(self):
        # Local coordinate system:
        # origin = center of vessel at its bottom
        # z = vessel axis

        z_bottom = 0.0
        z_top = self.base_thickness + self.height + self.cover_thickness

        z_base = openmc.ZPlane(z0=z_bottom)
        z_vessel_bottom = openmc.ZPlane(z0=self.base_thickness)
        z_vessel_top = openmc.ZPlane(
            z0=self.base_thickness + self.height
        )
        z_top = openmc.ZPlane(z0=z_top)

        inner = openmc.ZCylinder(r=self.radius)
        outer = openmc.ZCylinder(r=self.external_radius)

        base_region = (
            +z_base
            & -z_vessel_bottom
            & -outer
        )

        cylindrical_region = (
            +z_vessel_bottom
            & -z_vessel_top
            & +inner
            & -outer
        )

        cover_region = (
            +z_vessel_top
            & -z_top
            & -outer
        )

        region = (
            base_region
            | cylindrical_region
            | cover_region
        )

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


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
class Breeder(Component):
    density: float
    mass: float | None = None
    volume: float | None = None
    material: openmc.Material | None = None

    def region(self, vessel):
        fill_height = vessel.height_for_volume(
            self.requested_volume
        )

        bottom = openmc.ZPlane(z0=vessel.fill_bottom)
        top = openmc.ZPlane(
            z0=vessel.fill_bottom + fill_height
        )

        return (
            vessel.fill_region
            & +bottom
            & -top
        )

    @property
    def requested_volume(self):
        if self.volume is not None:
            return self.volume

        return self.mass / self.density

    def geometry(self, vessel):
        cell = openmc.Cell(
            name=self.name,
            region=self.region(vessel),
            fill=self.material,
        )

        return [cell]

@dataclass
class HeadSpace(Component):
    material: openmc.Material

    def region(self, vessel, breeder):
        return vessel.fill_region & ~breeder.region(vessel)

    def geometry(self, vessel, breeder):
        cell = openmc.Cell(
            name=self.name,
            region=self.region(vessel, breeder),
            fill=self.material,
        )

        return [cell]


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

class BreederVessel1LAssembly(Component):

    def geometry(self):

        vessel_region = self.vessel.fill_region

        breeder_region = self.breeder.region(
            vessel_region
        )

        headspace_region = (
            vessel_region
            & ~breeder_region
        )

        vessel_cells = self.vessel.geometry()

        breeder_cell = openmc.Cell(
            name=self.breeder.name,
            region=breeder_region,
            fill=self.breeder.material,
        )

        headspace_cell = openmc.Cell(
            name=self.headspace.name,
            region=headspace_region,
            fill=self.headspace.material,
        )

        return (
            vessel_cells
            + [breeder_cell, headspace_cell]
        )