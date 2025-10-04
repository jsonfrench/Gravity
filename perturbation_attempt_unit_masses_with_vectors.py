import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons

# === Setup Simulation ===
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

G = 6.67430e-11  # SI units

# Central + orbiting masses
central_mass = 100.0
m1 = 1.0
m2 = 1.0

# Orbital radii
r1 = 1.0
r2 = 2.0

# Circular velocity magnitudes
v1 = np.sqrt(G * central_mass / r1)
v2 = np.sqrt(G * central_mass / r2)

# Add particles
sim.add(m=central_mass)                  # index 0
sim.add(m=m1, x=r1, y=0, vy=v1)          # index 1 (Earth analogue)
sim.add(m=m2, x=r2, y=0, vy=v2)          # index 2 (Mars analogue)

# === Precompute simulation ===
t_max = 1e6   # total duration
dt = 1e3      # step size
times = np.arange(0, t_max, dt)

earth_positions_actual = []
mars_positions = []

sim_copy = sim.copy()
for t in times:
    sim_copy.integrate(t)
    p1, p2 = sim_copy.particles[1], sim_copy.particles[2]
    earth_positions_actual.append([p1.x, p1.y])
    mars_positions.append([p2.x, p2.y])

earth_positions_actual = np.array(earth_positions_actual)
mars_positions = np.array(mars_positions)

# === Matplotlib setup ===
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.3)

ax.plot(earth_positions_actual[:, 0], earth_positions_actual[:, 1],
        label='Object 1 (Earth analogue)', color='blue', alpha=0.6)
ax.plot(mars_positions[:, 0], mars_positions[:, 1],
        label='Object 2 (Mars analogue)', color='red', alpha=0.6)
ax.plot(0, 0, 'yo', label='Central Mass')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Perturbation Test: 100–1–1 System")
ax.set_aspect('equal')
ax.grid(True)
ax.legend()

# === Widgets ===
axbox = plt.axes([0.25, 0.15, 0.5, 0.05])
text_box = TextBox(axbox, f'Start Step (0-{len(times)-14}):', initial="0")

ax_radio = plt.axes([0.8, 0.15, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('3', '7', '14'))
selected_steps = [3]  # default

# === Update Function ===
def update(val):
    try:
        user_step = int(val)
        period_steps = selected_steps[0]

        if not (0 <= user_step <= (len(times) - period_steps)):
            print(f"Start step must be between 0 and {len(times) - period_steps}")
            return

        start_time = times[user_step]
        end_time = times[user_step + period_steps]

        # --- State at start (full sim) ---
        sim_full = sim.copy()
        sim_full.integrate(start_time)
        earth_init = sim_full.particles[1]

        # --- Mars-less simulation (central + Earth only) ---
        sim_theory = rebound.Simulation()
        sim_theory.units = ('s', 'm', 'kg')
        sim_theory.integrator = "ias15"
        sim_theory.add(m=central_mass)
        sim_theory.add(m=m1,
                       x=earth_init.x, y=earth_init.y,
                       vx=earth_init.vx, vy=earth_init.vy)

        sim_theory.integrate(end_time - start_time)
        earth_theory_end = np.array([sim_theory.particles[1].x,
                                     sim_theory.particles[1].y])

        # --- Actual full simulation end state ---
        sim_full.integrate(end_time)
        earth_end = sim_full.particles[1]
        mars_end = sim_full.particles[2]
        earth_actual_end = np.array([earth_end.x, earth_end.y])
        mars_actual_end = np.array([mars_end.x, mars_end.y])

        # --- Perturbation vector (displacement) ---
        perturbation_vector = earth_actual_end - earth_theory_end

        # --- Compute perturbations for last 3 steps (finite difference) ---
        perturbations = []
        for offset in [period_steps - 3, period_steps - 2, period_steps - 1]:
            if offset < 0:
                continue
            sim_copy = sim.copy()
            sim_copy.integrate(start_time + offset * dt)
            e_act = sim_copy.particles[1]
            actual_pos = np.array([e_act.x, e_act.y])

            sim_theory_copy = rebound.Simulation()
            sim_theory_copy.units = ('s', 'm', 'kg')
            sim_theory_copy.integrator = "ias15"
            sim_theory_copy.add(m=central_mass)
            sim_theory_copy.add(m=m1,
                                x=earth_init.x, y=earth_init.y,
                                vx=earth_init.vx, vy=earth_init.vy)
            sim_theory_copy.integrate(offset * dt)
            e_theory = sim_theory_copy.particles[1]
            theory_pos = np.array([e_theory.x, e_theory.y])
            perturbations.append(actual_pos - theory_pos)

        perturbations = np.array(perturbations)
        if len(perturbations) >= 3:
            perturbation_approx_vel = perturbations[2] - perturbations[1]
            perturbation_approx_accel = perturbations[2] - 2*perturbations[1] + perturbations[0]
        else:
            perturbation_approx_vel = np.array([0,0])
            perturbation_approx_accel = np.array([0,0])

        print("Perturbation displacement:", perturbation_vector)
        print("Approx perturbation vel:", perturbation_approx_vel)
        print("Approx perturbation accel:", perturbation_approx_accel)

        # --- Clear previous dynamic plots ---
        [l.remove() for l in ax.lines[3:]]
        [a.remove() for a in ax.patches]

        # --- Plot start/end and perturbations ---
        ax.plot(earth_init.x, earth_init.y, 'co', label=f'Start (step {user_step})')
        ax.plot(*earth_actual_end, 'bo', label='Actual End')
        ax.plot(*earth_theory_end, 'go', label='Theory End (Mars-less)')
        ax.plot(*mars_actual_end, 'ro', label='Mars End')

        # Arrows for perturbation, vel, accel
        scale = 0.5  # visual scaling
        def plot_arrow(vec, color, label):
            if np.linalg.norm(vec) > 0:
                unit = vec/np.linalg.norm(vec)
                arr = unit * scale
                ax.arrow(*earth_actual_end, arr[0], arr[1],
                         color=color, head_width=0.05, head_length=0.1, label=label)

        plot_arrow(perturbation_vector, 'magenta', 'Perturbation')
        plot_arrow(perturbation_approx_vel, 'red', 'Perturb Vel')
        plot_arrow(perturbation_approx_accel, 'yellow', 'Perturb Accel')

        # --- Refresh legend ---
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper right')
        fig.canvas.draw_idle()

    except Exception as e:
        print("Error:", e)

# Connect events
text_box.on_submit(update)
def on_radio_change(label):
    selected_steps[0] = int(label)
    update(text_box.text)

radio.on_clicked(on_radio_change)

plt.show()
