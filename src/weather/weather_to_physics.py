"""
Weather → Physics Bridge Module.

THIS IS THE BRIDGE between ML perception and the mechanical physics engine.

Converts detected weather conditions into physics-relevant parameters that
feed directly into the vehicle dynamics simulator:
    - Road surface type → tire_model.py friction coefficients
    - Reaction time penalty → AEB controller threshold adjustment
    - Visibility range → maximum effective detection distance

Integration points:
    - weather_to_surface() output maps to tire_model.SURFACES keys ("dry", "wet")
    - reaction_penalty adds to driver's base reaction time in AEB controller
    - visibility_range limits the maximum distance for TTC calculation
"""


def weather_to_surface(weather_class):
    """
    Convert detected weather → road surface friction coefficient name.

    Maps directly to the surface names defined in
    src/vehicle_dynamics/tire_model.py → SURFACES dict.

    Mapping:
        Clear → "dry"  (μ=0.85)
        Rain  → "wet"  (μ=0.55)
        Fog   → "wet"  (μ=0.55, also visibility reduced)
        Night → "dry"  (μ=0.85, road is dry but reaction time increases)

    Args:
        weather_class: str — one of "Clear", "Rain", "Fog", "Night"

    Returns:
        surface: str — matches tire_model.py SURFACES keys
    """
    mapping = {
        "Clear": "dry",
        "Rain": "wet",
        "Fog": "wet",
        "Night": "dry",
    }
    return mapping.get(weather_class, "dry")


def weather_to_reaction_penalty(weather_class):
    """
    Additional reaction time penalty based on weather conditions.

    This penalty ADDS to the driver's base reaction time (from alertness state).
    It accounts for the cognitive load of adverse conditions:

        Clear → +0.0s  (no additional burden)
        Rain  → +0.3s  (wipers, reduced visibility, wet road anxiety)
        Fog   → +0.5s  (severely reduced visibility, slower decision-making)
        Night → +0.2s  (reduced contrast, harder to see pedestrians)

    Example:
        Driver is ALERT (base delay = 0.7s) + Rain (+0.3s)
        → Total reaction time = 1.0s
        → AEB triggers at TTC = 1.5 + 1.0 = 2.5s

    Args:
        weather_class: str — one of "Clear", "Rain", "Fog", "Night"

    Returns:
        penalty: float — seconds to add to base reaction time
    """
    penalties = {
        "Clear": 0.0,
        "Rain": 0.3,
        "Fog": 0.5,
        "Night": 0.2,
    }
    return penalties.get(weather_class, 0.0)


def get_physics_params(weather_class):
    """
    Get all physics-relevant parameters from weather classification.

    This is the primary interface for the vehicle dynamics simulator —
    call this once per frame to get all weather-dependent physics parameters.

    Args:
        weather_class: str — one of "Clear", "Rain", "Fog", "Night"

    Returns:
        dict with:
            "surface": str — for tire_model.py (determines friction coefficient)
            "reaction_penalty": float — seconds to add to driver reaction time
            "visibility_range": float — meters, maximum effective detection distance
    """
    visibility = {
        "Clear": 100.0,   # Full range, limited by camera/sensor capability
        "Rain": 60.0,     # Reduced by rain droplets, spray, reflections
        "Fog": 30.0,      # Severely reduced, dense fog can be < 30m
        "Night": 50.0,    # Limited by headlight range and contrast
    }

    return {
        "surface": weather_to_surface(weather_class),
        "reaction_penalty": weather_to_reaction_penalty(weather_class),
        "visibility_range": visibility.get(weather_class, 100.0),
    }
