"""
Vehicle Front-End Geometry Parameters.

Defines the physical geometry of the vehicle front that affects
pedestrian impact kinematics and injury patterns.

These parameters determine:
- Where on the vehicle the pedestrian's head strikes (Wrap-Around Distance)
- The acceleration pulse shape during head impact
- Energy absorption by the crumple zone

Reference: Euro NCAP Pedestrian Testing Protocol v9.0
"""

from dataclasses import dataclass


@dataclass
class VehicleFrontGeometry:
    """Physical dimensions of the vehicle front-end for impact analysis."""

    # ─── Heights from Ground ──────────────────────────────────
    bumper_bottom_height: float = 0.350     # meters
    bumper_top_height: float = 0.500        # meters
    hood_leading_edge_height: float = 0.700 # meters (bonnet leading edge — BLE)
    hood_rear_height: float = 0.900         # meters (base of windshield)
    windshield_base_height: float = 1.050   # meters
    windshield_top_height: float = 1.450    # meters
    roof_height: float = 1.470              # meters

    # ─── Horizontal Dimensions ────────────────────────────────
    bumper_depth: float = 0.100             # meters (front-to-back thickness)
    hood_length: float = 0.900             # meters (horizontal length of hood)
    bumper_to_hood_le: float = 0.150       # meters (horizontal offset)

    # ─── Angles ───────────────────────────────────────────────
    hood_angle_deg: float = 10.0           # degrees from horizontal
    windshield_angle_deg: float = 30.0     # degrees from vertical

    # ─── Stiffness / Deformation ──────────────────────────────
    bumper_stiffness: float = 200e3        # N/m (bumper system stiffness)
    hood_stiffness: float = 50e3           # N/m (hood panel stiffness)
    hood_deformation_depth: float = 0.080  # meters (max crumple zone depth)
    windshield_stiffness: float = 300e3    # N/m (laminated glass)

    # ─── Vehicle Width ────────────────────────────────────────
    vehicle_width: float = 1.800           # meters

    def impact_zone(self, wrap_around_distance):
        """
        Determine which part of the vehicle the pedestrian's head hits.

        Args:
            wrap_around_distance: float (meters, measured along vehicle surface)

        Returns:
            str: "bumper", "hood_front", "hood_mid", "hood_rear", "windshield", "roof"
        """
        if wrap_around_distance < self.bumper_top_height:
            return "bumper"
        elif wrap_around_distance < self.hood_leading_edge_height + self.hood_length * 0.33:
            return "hood_front"
        elif wrap_around_distance < self.hood_leading_edge_height + self.hood_length * 0.66:
            return "hood_mid"
        elif wrap_around_distance < self.windshield_base_height:
            return "hood_rear"
        elif wrap_around_distance < self.windshield_top_height:
            return "windshield"
        else:
            return "roof"

    def stiffness_at_zone(self, zone):
        """
        Get the local stiffness at the impact zone.
        Lower stiffness = more deformation = less severe head injury.

        Returns:
            stiffness: float (N/m)
        """
        stiffness_map = {
            "bumper": self.bumper_stiffness,
            "hood_front": self.hood_stiffness * 1.2,  # stiffer near edges
            "hood_mid": self.hood_stiffness,            # softest zone
            "hood_rear": self.hood_stiffness * 1.5,     # near hinges
            "windshield": self.windshield_stiffness,
            "roof": self.windshield_stiffness * 1.5,
        }
        return stiffness_map.get(zone, self.hood_stiffness)


# ─── Preset Vehicle Geometries ────────────────────────────────

SEDAN_GEOMETRY = VehicleFrontGeometry()

SUV_GEOMETRY = VehicleFrontGeometry(
    bumper_bottom_height=0.450,
    bumper_top_height=0.600,
    hood_leading_edge_height=0.850,
    hood_rear_height=1.050,
    windshield_base_height=1.200,
    windshield_top_height=1.600,
    roof_height=1.750,
)

COMPACT_GEOMETRY = VehicleFrontGeometry(
    bumper_bottom_height=0.300,
    bumper_top_height=0.450,
    hood_leading_edge_height=0.650,
    hood_rear_height=0.850,
    windshield_base_height=0.950,
    windshield_top_height=1.350,
    roof_height=1.400,
)
