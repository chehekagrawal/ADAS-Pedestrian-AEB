import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.vehicle_dynamics.thermodynamics import BrakeThermodynamics
from src.vehicle_dynamics.dynamics_sim import braking_distance

def plot_temperature_and_fade():
    """
    Simulates a spirited driving session with multiple hard braking events
    to visibly demonstrate temperature spikes and friction loss.
    """
    thermo = BrakeThermodynamics()
    
    times = []
    temps = []
    frictions = []
    
    current_time = 0.0
    dt = 0.1
    
    # Driving scenario:
    # 0-60s: Drive at 100kmh, cool brakes
    # 60s: Hard stop 100 -> 0kmh
    # 65s-120s: Accelerate and drive at 120kmh
    # 120s: Hard stop 120 -> 0kmh
    # 125s-180s: Drive at 120kmh
    # 180s: Hard stop 120 -> 0kmh (now very hot)
    # 185s-300s: High speed coasting to cool down
    
    events = [
        (60, 100, 0),
        (120, 120, 0),
        (180, 120, 0),
        (240, 150, 0) # massive stop
    ]
    
    speed = 100.0
    
    for _ in range(int(350 / dt)):
        # Check for events
        for ev_time, v_start, v_end in events:
            if abs(current_time - ev_time) < (dt / 2):
                thermo.apply_braking_event(v_start, v_end)
                speed = v_end
                break
        else:
            # Not braking, maybe accelerating back up
            if speed < 120 and current_time > 65:
                speed += 5.0 * dt # recover speed
            thermo.cool_down(dt, speed)
        
        times.append(current_time)
        temps.append(thermo.current_temperature)
        frictions.append(thermo.get_friction_multiplier() * 100.0)
        
        current_time += dt

    # ── Plots ──
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[2, 1])
    
    # 1. Temperature Plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, temps, color='#cc0000', linewidth=2.5, label="Rotor Temperature")
    ax1.axhline(300, color='orange', linestyle='--', alpha=0.7, label="Fade Onset (300°C)")
    ax1.axhline(450, color='red', linestyle='--', alpha=0.7, label="Critical Fade (450°C)")
    ax1.set_ylabel("Rotor Temperature (°C)", fontsize=12)
    ax1.set_title("Brake Fluid & Rotor Thermal Dynamics (Multiple AEB Events)", fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    
    # Annotate braking events
    ax1.annotate('100 → 0 km/h', xy=(60, temps[int(60/dt)]), xytext=(40, temps[int(60/dt)]+100),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    ax1.annotate('120 → 0 km/h', xy=(180, temps[int(180/dt)]), xytext=(160, temps[int(180/dt)]+100),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    ax1.annotate('150 → 0 km/h', xy=(240, temps[int(240/dt)]), xytext=(220, temps[int(240/dt)]+100),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    # 2. Friction Plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(times, frictions, color='#0066cc', linewidth=2.5, label="Effective Grip (%)")
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Friction Coef. %", fontsize=12)
    ax2.fill_between(times, 0, frictions, color='#0066cc', alpha=0.2)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    
    os.makedirs("results/vehicle_dynamics", exist_ok=True)
    plt.savefig("results/vehicle_dynamics/thermodynamic_fade.png", dpi=300, bbox_inches='tight')
    print("Saved plots to results/vehicle_dynamics/thermodynamic_fade.png")

def plot_stopping_distance_increase():
    """
    Shows how AEB stopping distance balloons out when brakes are hot.
    """
    speeds = np.linspace(30, 120, 50)
    
    thermo = BrakeThermodynamics()
    cold_dists = []
    hot_dists = []
    
    for s in speeds:
        base_dist = braking_distance(s, "dry")
        cold_dists.append(base_dist)
        
        # Simulate hot brakes
        thermo.current_temperature = 550 # Severe fade
        friction_mult = thermo.get_friction_multiplier()
        # Distance = v^2 / (2 * a * friction)
        hot_dists.append(base_dist / friction_mult)

    plt.figure(figsize=(10, 6))
    plt.plot(speeds, cold_dists, 'g-', linewidth=3, label="Cold Brakes (20°C)")
    plt.plot(speeds, hot_dists, 'r-', linewidth=3, label="Hot Brakes (550°C)")
    plt.fill_between(speeds, cold_dists, hot_dists, color='red', alpha=0.1, label="Danger Zone")
    
    plt.title("AEB Stopping Distance: Cold vs Hot Brakes", fontsize=14)
    plt.xlabel("Vehicle Speed (km/h)", fontsize=12)
    plt.ylabel("Stopping Distance (meters)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    
    os.makedirs("results/vehicle_dynamics", exist_ok=True)
    plt.savefig("results/vehicle_dynamics/hot_brakes_stopping_distance.png", dpi=300)
    print("Saved plots to results/vehicle_dynamics/hot_brakes_stopping_distance.png")

if __name__ == "__main__":
    plot_temperature_and_fade()
    plot_stopping_distance_increase()
