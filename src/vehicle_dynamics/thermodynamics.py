import numpy as np
import math

class BrakeThermodynamics:
    """
    Models the thermodynamic heating and cooling of cast-iron brake rotors.
    Calculates temperature spikes from kinetic energy conversion and the 
    subsequent 'brake fade' (loss of friction) at high temperatures.
    """
    def __init__(self, 
                 mass_kg=1500.0, 
                 rotor_mass_kg=40.0, 
                 ambient_temp=20.0, 
                 cp_cast_iron=450.0): # J/(kg*K)
        """
        Args:
            mass_kg: Vehicle mass in kg.
            rotor_mass_kg: Total mass of all 4 brake rotors in kg (approx 10kg each).
            ambient_temp: Starting and resting temperature in Celsius.
            cp_cast_iron: Specific heat capacity of gray cast iron in J/(kg*K).
        """
        self.mass = mass_kg
        self.rotor_mass = rotor_mass_kg
        self.ambient_temp = ambient_temp
        self.cp = cp_cast_iron
        
        self.current_temperature = self.ambient_temp
        
        # Convective cooling constant (estimated based on typical airflow at speed)
        self.cooling_coefficient_base = 0.0001
        self.cooling_speed_factor = 0.00005

    def apply_braking_event(self, speed_start_kmh, speed_end_kmh):
        """
        Calculates the massive heat spike from a braking event.
        Converts 100% of kinetic energy lost directly to thermal energy in rotors.
        """
        v_start_ms = speed_start_kmh / 3.6
        v_end_ms = speed_end_kmh / 3.6
        
        # Kinetic energy dissipated
        ke_start = 0.5 * self.mass * (v_start_ms ** 2)
        ke_end = 0.5 * self.mass * (v_end_ms ** 2)
        energy_joules = max(0.0, ke_start - ke_end)
        
        # Delta T = Q / (m * c)
        delta_temp = energy_joules / (self.rotor_mass * self.cp)
        self.current_temperature += delta_temp
        return self.current_temperature

    def apply_continuous_braking(self, deceleration_ms2, speed_kmh, dt):
        """
        Alternative: apply braking continuously per frame (dt).
        """
        # Power = Force * Velocity = (m * a) * v
        if deceleration_ms2 <= 0:
            return self.current_temperature
            
        v_ms = speed_kmh / 3.6
        braking_force = self.mass * deceleration_ms2
        energy_joules = braking_force * v_ms * dt
        
        delta_temp = energy_joules / (self.rotor_mass * self.cp)
        self.current_temperature += delta_temp
        return self.current_temperature

    def cool_down(self, dt, current_speed_kmh):
        """
        Newton's Law of Cooling applied per time-step (dt).
        Cooling rate increases with vehicle speed due to forced convection.
        """
        v_ms = current_speed_kmh / 3.6
        # Dynamic cooling rate based on airflow
        k = self.cooling_coefficient_base + (self.cooling_speed_factor * v_ms)
        
        # T(t) = T_env + (T_0 - T_env) * exp(-k * dt)
        temp_diff = self.current_temperature - self.ambient_temp
        self.current_temperature = self.ambient_temp + (temp_diff * math.exp(-k * dt))
        return self.current_temperature

    def get_friction_multiplier(self):
        """
        Returns a multiplier [0.0 to 1.0] representing brake fade.
        Cast iron brakes work fine up to ~300C, fade rapidly between 400C-600C.
        """
        t = self.current_temperature
        
        if t < 300:
            return 1.0
        elif t < 450:
            # Linear drop from 1.0 down to 0.7
            return 1.0 - 0.3 * ((t - 300) / 150)
        elif t < 650:
            # Steep drop from 0.7 down to 0.3
            return 0.7 - 0.4 * ((t - 450) / 200)
        else:
            # Catastrophic fade
            return max(0.1, 0.3 - 0.2 * ((t - 650) / 200))

    def get_aeb_buffer_penalty(self):
        """
        Translates real-time brake fade into an AEB response penalty.
        If brakes are faded, AEB must trigger earlier (more TTC needed) to avoid collision.
        Returns: additional TTC seconds needed.
        """
        multiplier = self.get_friction_multiplier()
        if multiplier >= 0.95:
            return 0.0
            
        # If friction is 0.5, stopping distance doubles. 
        # So we penalize AEB trigger threshold severely.
        # e.g., multiplier 0.5 -> penalty 1.0 seconds
        penalty = 0.5 * (1.0 / multiplier - 1.0)
        
        # Max penalty constraint
        return min(3.0, penalty)
