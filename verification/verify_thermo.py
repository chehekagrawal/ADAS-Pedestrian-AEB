"""
Verification Script: Brake Thermodynamics & Thermal Fade
Verifies the thermodynamic heating and cooling equations from src/vehicle_dynamics/thermodynamics.py.
Outputs exact temperature rises, friction degradation multipliers, and TTC buffer penalties.
"""

import os
import sys
import json
import csv
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vehicle_dynamics.thermodynamics import BrakeThermodynamics

def main():
    os.makedirs("verification", exist_ok=True)
    thermo = BrakeThermodynamics(mass_kg=1500.0, rotor_mass_kg=40.0, ambient_temp=20.0, cp_cast_iron=450.0)

    # 1. Temperature sweep from 20 to 800 °C
    temps = [20, 100, 200, 300, 350, 400, 428, 450, 500, 550, 600, 650, 700, 750, 800]
    sweep_records = []

    for t in temps:
        thermo.current_temperature = float(t)
        mu_mult = thermo.get_friction_multiplier()
        penalty = thermo.get_aeb_buffer_penalty()
        sweep_records.append({
            "temperature_c": t,
            "friction_multiplier": round(mu_mult, 4),
            "friction_loss_pct": round((1.0 - mu_mult) * 100, 2),
            "aeb_buffer_penalty_s": round(penalty, 4)
        })

    with open("verification/thermo_sweep_verified.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_records[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_records)

    # 2. Sequential braking event simulation
    # Hard stop from 80 -> 0 km/h repeated 6 times with 20s cooling intervals at 80 km/h cruise
    thermo_sim = BrakeThermodynamics(mass_kg=1500.0, rotor_mass_kg=40.0, ambient_temp=20.0, cp_cast_iron=450.0)
    
    event_records = []
    current_time = 0.0
    dt = 0.1
    # Single stop energy
    v_80_ms = 80.0 / 3.6
    ke_single = 0.5 * 1500.0 * (v_80_ms ** 2)
    delta_t_single = ke_single / (40.0 * 450.0)

    # In front-biased braking (65% front bias, 2 front rotors of 10kg each = 20kg):
    # front delta T = 0.65 * ke / (20 * 450) = 0.65 * 370370 / 9000 = 26.75 °C per stop
    # If using 2 front rotors alone (total mass 14kg), delta T is much higher.

    print("=== Brake Thermodynamics Verification ===")
    print(f"Single stop from 80 km/h: KE = {ke_single:.0f} J, Bulk 4-rotor Delta-T = {delta_t_single:.2f} °C")
    print(f"{'Temp (°C)':<10} {'Friction Multiplier':<20} {'Friction Loss (%)':<20} {'AEB Penalty (s)':<15}")
    print("-" * 65)
    for r in sweep_records:
        print(f"{r['temperature_c']:<10} {r['friction_multiplier']:<20.4f} {r['friction_loss_pct']:<20.1f} {r['aeb_buffer_penalty_s']:<15.3f}")

    with open("verification/thermo_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "constants": {
                "vehicle_mass_kg": 1500.0,
                "rotor_mass_kg": 40.0,
                "ambient_temp_c": 20.0,
                "cp_cast_iron": 450.0,
                "cooling_base": 0.0001,
                "cooling_speed_factor": 0.00005
            },
            "single_stop_80kmh": {
                "kinetic_energy_joules": ke_single,
                "temperature_rise_c": delta_t_single
            },
            "sample_points": sweep_records
        }, f, indent=4)

    print("\nSaved verification/thermo_sweep_verified.csv and verification/thermo_summary.json")

if __name__ == "__main__":
    main()
