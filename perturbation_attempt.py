import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons

# Constants
YEAR_DAYS = 3 * 365
DAY_SEC = 24 * 60 * 60

# === 1. FULL SIMULATION: Sun + Earth + Mars ===
sim_full = rebound.Simulation()
sim_full.units = ('s', 'm', 'kg')
sim_full.integrator = "ias15"

# Add Sun, Earth, Mars
sim_full.add(m=1.989e30)  # Sun
sim_full.add(m=5.972e24, a=1.496e11, e=0.0167)  # Earth
sim_full.add(m=6.417e23, a=2.279e11, e=0.0934)  # Mars

# Run full simulation
times = np.arange(0, YEAR_DAYS * DAY_SEC, DAY_SEC)
earth_positions_actual = []
mars_positions = []

for t in times:
    sim_full.integrate(t)
    earth = sim_full.particles[1]
    mars = sim_full.particles[2]
    earth_positions_actual.append([earth.x, earth.y])
    mars_positions.append([mars.x, mars.y])

earth_positions_actual = np.array(earth_positions_actual)
mars_positions = np.array(mars_positions)

# === 2. Matplotlib Plot and Widgets ===
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.3)  # Leave space for textbox and buttons

# Static plots
ax.plot(earth_positions_actual[:, 0], earth_positions_actual[:, 1], label='Earth (actual)', color='blue', alpha=0.6)
ax.plot(mars_positions[:, 0], mars_positions[:, 1], label='Mars', color='red', alpha=0.6)
ax.plot(0, 0, 'yo', label='Sun')
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.set_title("Earth's Orbit: Actual vs Theoretical")
ax.set_aspect('equal')
ax.grid(True)
legend = ax.legend()

# Textbox for day input
axbox = plt.axes([0.25, 0.15, 0.5, 0.05])
text_box = TextBox(axbox, f'Start Day (0-{YEAR_DAYS-14}):', initial="0")

# Buttons for selecting period
ax_radio = plt.axes([0.8, 0.15, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('3', '7', '14'))
selected_days = [3]  # Mutable default for closure

# Plot update function
def update(val):
    try:
        user_day = int(val)
        period_days = selected_days[0]

        if not (0 <= user_day <= (YEAR_DAYS - period_days)):
            print(f"Start day must be between 0 and {YEAR_DAYS - period_days}")
            return

        start_time = user_day * DAY_SEC
        end_time = start_time + period_days * DAY_SEC

        # Get Earth state at user-specified time
        sim_copy = sim_full.copy()
        sim_copy.integrate(start_time)
        earth_init = sim_copy.particles[1]

        # === Sun + Earth only simulation ===
        sim_theory = rebound.Simulation()
        sim_theory.units = ('s', 'm', 'kg')
        sim_theory.add(m=1.989e30)
        sim_theory.add(m=5.972e24,
                       x=earth_init.x, y=earth_init.y,
                       vx=earth_init.vx, vy=earth_init.vy, vz=earth_init.vz)

        times_short = np.arange(0, period_days * DAY_SEC, DAY_SEC)
        earth_theory_positions = []
        sim_theory.integrate(period_days * DAY_SEC)
        theory_pos_end = np.array([sim_theory.particles[1].x, sim_theory.particles[1].y])

        # let's try to retrieve the acceleration of the perturbations
        for t in times_short:
            sim_theory.integrate(t)
            e = sim_theory.particles[1]
            earth_theory_positions.append([e.x, e.y])
        
        earth_theory_positions = np.array(earth_theory_positions)

        # perturbations = np.zeros(3)
        # i = 0
        # while i < 3:
        #     perturbations[i] = earth_positions_actual[user_day + period_days -i, :] - earth_theory_positions[-1-i, :]
        #     i = i + 1



        # Clear previous dynamic plot elements
        [l.remove() for l in ax.lines[3:]]  # Keep initial 3 plots

        # Recompute final positions from full sim
        sim_copy = sim_full.copy()
        sim_copy.integrate(end_time)
        earth_end = sim_copy.particles[1]
        mars_end = sim_copy.particles[2]

        actual_pos_start = np.array([earth_init.x, earth_init.y])
        actual_pos_end = np.array([earth_end.x, earth_end.y])
        mars_pos_end = np.array([mars_end.x, mars_end.y])

        perturbation_vector = actual_pos_end - theory_pos_end

                # === Compute perturbations for last 3 days ===
        perturbations = []

        for offset in [period_days - 3, period_days - 2, period_days - 1]:
            if offset < 0:
                print("Simulation period too short for second-order difference.")
                return

            # Actual position
            sim_copy = sim_full.copy()
            sim_copy.integrate(start_time + offset * DAY_SEC)
            earth_actual = sim_copy.particles[1]
            actual_pos = np.array([earth_actual.x, earth_actual.y])

            # Theoretical position
            sim_theory_copy = rebound.Simulation()
            sim_theory_copy.units = ('s', 'm', 'kg')
            sim_theory_copy.add(m=1.989e30)
            sim_theory_copy.add(m=5.972e24,
                                x=earth_init.x, y=earth_init.y,
                                vx=earth_init.vx, vy=earth_init.vy, vz=earth_init.vz)
            sim_theory_copy.integrate(offset * DAY_SEC)
            earth_theory = sim_theory_copy.particles[1]
            theory_pos = np.array([earth_theory.x, earth_theory.y])

            perturbation = actual_pos - theory_pos
            perturbations.append(perturbation)

        # Compute second-order difference assuming dt = 1 day?
        perturbation_approx_accel = perturbations[2] - 2 * perturbations[1] + perturbations[0]
        perturbation_approx_vel = perturbations[2] - perturbations[1]

        # Print result in console
        print("Approximate second-order perturbation (acceleration):", perturbation_approx_accel)


        # Plot dynamic elements
        ax.plot(*actual_pos_start, 'co', label=f'Earth Start (Day {user_day})')
        ax.plot(*actual_pos_end, 'bo', label=f'Earth Actual ({period_days}d)')
        ax.plot(*theory_pos_end, 'go', label=f'Theoretical ({period_days}d)')
        ax.plot(*mars_pos_end, 'ro', label='Mars')

        # Normalize and scale
        scale_length = 5e10  # in meters, tweak this for visibility
        unit_vector = perturbation_vector / np.linalg.norm(perturbation_vector)
        arrow_vector = unit_vector * scale_length

        ax.arrow(*actual_pos_end,
                *arrow_vector,
                color='magenta', head_width=1e9, head_length=2e9,
                label='Perturbation Vector')
        
        # Normalize and scale velocity of perturbation 
        unit_vector_vel = perturbation_approx_vel / np.linalg.norm(perturbation_approx_vel)
        arrow_vector_vel = unit_vector_vel * scale_length

        ax.arrow(*actual_pos_end,
                *arrow_vector_vel,
                color='red', head_width=1e9, head_length=2e9,
                label='Perturbation Vector (vel)')
        
        # Normalize and scale acceleration of perturbation 
        unit_vector_accel = perturbation_approx_accel / np.linalg.norm(perturbation_approx_accel)
        arrow_vector_accel = unit_vector_accel * scale_length

        ax.arrow(*actual_pos_end,
                *arrow_vector_accel,
                color='yellow', head_width=1e9, head_length=2e9,
                label='Perturbation Vector (accel)')

        
        # Rebuild legend (only current items)
        handles, labels = ax.get_legend_handles_labels()

        # Use dictionary to remove duplicate labels (like 'Perturbation Vector')
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper right')

        fig.canvas.draw_idle()

    except Exception as e:
        print("Error:", e)

# Connect TextBox
text_box.on_submit(update)

# Handle period switch
def on_radio_change(label):
    selected_days[0] = int(label)
    # Re-run current input with new period
    update(text_box.text)

radio.on_clicked(on_radio_change)

plt.show()
