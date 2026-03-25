"""
Vehicle Parameters — Real-world specifications for a typical C-segment sedan.

All values are based on published automotive data and engineering references.
These parameters feed into the vehicle dynamics simulator, tire model,
and weight transfer calculations.

Reference: Rajamani, R. "Vehicle Dynamics and Control" (Springer)
"""

from dataclasses import dataclass, field


@dataclass
class VehicleParameters:
    """Complete vehicle specification for dynamics simulation."""

    # ─── Mass & Inertia ───────────────────────────────────────────
    mass: float = 1500.0            # kg (curb weight, typical C-segment sedan)
    mass_distribution_front: float = 0.58   # 58% front (FWD vehicle)

    # ─── Geometry ─────────────────────────────────────────────────
    wheelbase: float = 2.67         # meters (front axle to rear axle)
    cg_height: float = 0.54        # meters (center of gravity height above ground)
    cg_to_front: float = 1.12      # meters (CG to front axle)
    cg_to_rear: float = 1.55       # meters (CG to rear axle = wheelbase - cg_to_front)
    track_width: float = 1.53      # meters (distance between left and right wheels)
    vehicle_length: float = 4.50   # meters (overall length)
    vehicle_width: float = 1.80    # meters (overall width)

    # ─── Aerodynamics ─────────────────────────────────────────────
    frontal_area: float = 2.2      # m² (cross-section area)
    drag_coeff: float = 0.30       # Cd (drag coefficient)
    air_density: float = 1.225     # kg/m³ (sea level, 15°C)

    # ─── Tires (205/55R16) ────────────────────────────────────────
    tire_radius: float = 0.316     # meters (rolling radius)
    tire_width: float = 0.205      # meters (tread width)
    rolling_resistance: float = 0.015   # Crr (rolling resistance coefficient)

    # ─── Brake System ─────────────────────────────────────────────
    # Front brakes (ventilated disc)
    brake_disc_diameter_front: float = 0.300    # meters
    brake_disc_eff_radius_front: float = 0.130  # meters (effective friction radius)
    pad_friction_coeff: float = 0.40            # μ_pad (semi-metallic compound)
    caliper_piston_area_front: float = 0.003    # m² (total piston area per caliper)

    # Rear brakes (solid disc)
    brake_disc_diameter_rear: float = 0.280     # meters
    brake_disc_eff_radius_rear: float = 0.115   # meters
    caliper_piston_area_rear: float = 0.0020    # m²

    # Hydraulic system
    max_brake_pressure: float = 120e5   # Pa (120 bar = 12 MPa)
    brake_bias_front: float = 0.65      # 65% of braking force goes to front

    # ─── ABS Parameters ───────────────────────────────────────────
    abs_enabled: bool = True
    abs_slip_threshold: float = 0.15    # Slip ratio at which ABS releases (15%)
    abs_slip_target: float = 0.10       # Target slip ratio during ABS (10%)
    abs_cycle_time: float = 0.050       # seconds per ABS cycle (50 ms)
    abs_pressure_release_rate: float = 0.7  # fraction of pressure released per cycle
    abs_pressure_buildup_rate: float = 1.2  # rate of pressure reapply

    # ─── Constants ────────────────────────────────────────────────
    gravity: float = 9.81          # m/s²

    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Ensure CG distances are consistent with wheelbase
        assert abs(self.cg_to_front + self.cg_to_rear - self.wheelbase) < 0.01, \
            f"CG distances ({self.cg_to_front} + {self.cg_to_rear}) must equal wheelbase ({self.wheelbase})"

    @property
    def static_front_load(self):
        """Static normal force on front axle (N)."""
        return self.mass * self.gravity * self.cg_to_rear / self.wheelbase

    @property
    def static_rear_load(self):
        """Static normal force on rear axle (N)."""
        return self.mass * self.gravity * self.cg_to_front / self.wheelbase

    @property
    def max_brake_force_front(self):
        """Maximum hydraulic brake force at front calipers (N)."""
        # Force = pressure × piston_area × pad_friction × (eff_radius / tire_radius)
        # Two front calipers
        return (2 * self.max_brake_pressure * self.caliper_piston_area_front *
                self.pad_friction_coeff * self.brake_disc_eff_radius_front / self.tire_radius)

    @property
    def max_brake_force_rear(self):
        """Maximum hydraulic brake force at rear calipers (N)."""
        return (2 * self.max_brake_pressure * self.caliper_piston_area_rear *
                self.pad_friction_coeff * self.brake_disc_eff_radius_rear / self.tire_radius)


# ─── Preset Vehicles ──────────────────────────────────────────────

SEDAN_DEFAULT = VehicleParameters()

SUV_PARAMS = VehicleParameters(
    mass=1900.0,
    cg_height=0.68,
    frontal_area=2.8,
    drag_coeff=0.35,
    tire_radius=0.360,
    rolling_resistance=0.018,
)

COMPACT_PARAMS = VehicleParameters(
    mass=1200.0,
    wheelbase=2.50,
    cg_height=0.48,
    cg_to_front=1.05,
    cg_to_rear=1.45,
    frontal_area=2.0,
    drag_coeff=0.28,
)
