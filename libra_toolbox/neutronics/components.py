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
class Firebrick(Component):
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
    """Breeder material filling the interior of a Vessel.

    The Breeder amount can be specified either by mass or by volume.
    The Vessel is responsible for defining the available internal
    volume and accounting for internal structures/obstructions.

    Parameters
    ----------
    density : float
        Breeder density [g/cm3].
    mass : float, optional
        Total Breeder mass [g].
    volume : float, optional
        Total Breeder volume [cm3].
    material : openmc.Material, optional
        OpenMC material used to fill the Breeder region.
    fill_direction : str
        Direction in which the material fills the vessel.
        Currently only ``"z"`` is supported.
    """

    density: float
    mass: Optional[float] = None
    volume: Optional[float] = None
    material: Optional[openmc.Material] = None
    fill_direction: str = "z"

    def __post_init__(self):
        if self.mass is None and self.volume is None:
            raise ValueError(
                "Breeder requires either 'mass' or 'volume'."
            )

        if self.mass is not None and self.volume is not None:
            raise ValueError(
                "Specify either 'mass' or 'volume', not both."
            )

        if self.density <= 0:
            raise ValueError("Breeder density must be positive.")

        if self.mass is not None and self.mass <= 0:
            raise ValueError("Breeder mass must be positive.")

        if self.volume is not None and self.volume <= 0:
            raise ValueError("Breeder volume must be positive.")

        if self.fill_direction != "z":
            raise NotImplementedError(
                "Only fill_direction='z' is currently supported."
            )

    @property
    def requested_volume(self) -> float:
        """Requested Breeder volume [cm3]."""
        if self.volume is not None:
            return self.volume

        return self.mass / self.density

    def geometry(self, vessel):
        """Build the Breeder OpenMC geometry inside ``vessel``.

        Parameters
        ----------
        vessel : Vessel
            Vessel containing the Breeder.

        Returns
        -------
        list[openmc.Cell]
            OpenMC cells representing the Breeder.
        """

        if self.material is None:
            raise ValueError(
                "Breeder requires an OpenMC material."
            )

        # Ask the vessel to determine the height required to contain
        # the requested volume while respecting its internal geometry.
        fill_height = vessel.height_for_volume(
            self.requested_volume
        )

        # Vessel provides the actual available internal region,
        # already excluding walls and internal structures.
        fill_region = vessel.fill_region

        # Restrict the fill to the required height.
        z_bottom = vessel.fill_bottom
        z_top = z_bottom + fill_height

        bottom = openmc.ZPlane(z0=z_bottom)
        top = openmc.ZPlane(z0=z_top)

        region = (
            fill_region
            & +bottom
            & -top
        )

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]

@dataclass
class HeadSpace(Component):
    """Head-space region inside a Vessel not occupied by the Breeder.

    By default, the HeadSpace fills all of the available vessel volume
    left unoccupied by the Breeder.

    Parameters
    ----------
    material : openmc.Material
        Material filling the head-space region, e.g. helium.
    """

    material: openmc.Material

    def geometry(self, vessel, breeder):
        """Build the head-space geometry inside ``vessel``.

        Parameters
        ----------
        vessel : Vessel
            Vessel containing the Breeder and head space.
        breeder : Breeder
            Breeder component occupying part of the vessel volume.

        Returns
        -------
        list[openmc.Cell]
            OpenMC cells representing the head space.
        """

        # Entire volume available for contents of the vessel.
        vessel_region = vessel.fill_region

        # Region occupied by the Breeder.
        breeder_cells = breeder.geometry(vessel)

        if len(breeder_cells) != 1:
            raise ValueError(
                "HeadSpace currently expects Breeder to produce "
                "exactly one OpenMC cell."
            )

        breeder_region = breeder_cells[0].region

        # Everything inside the vessel that is not the Breeder.
        headspace_region = vessel_region & ~breeder_region

        cell = openmc.Cell(
            name=self.name,
            region=headspace_region,
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