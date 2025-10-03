import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# === Setup Simulation ===
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

G = 6.67430e-11  # SI units

# Central mass
central_mass = 100.0
sim.add(m=central_mass)  # index 0

# Orbital parameters
r1 = 1.0
r2 = 2.0
m1 = m2 = 1.0

v1 = np.sqrt(G * central_mass / r1)
v2 = np.sqrt(G * central_mass / r2)

# Add orbiting masses
sim.add(m=m1, x=r1, y=0, vy=v1)  # index 1
sim.add(m=m2, x=r2, y=0, vy=v2)  # index 2

# === Precompute Orbits ===
t_max = 1e6  # total simulation time in seconds
n_steps = 1000
times = np.linspace(0, t_max, n_steps)

positions_0 = np.zeros((n_steps, 2))
positions_1 = np.zeros((n_steps, 2))
positions_2 = np.zeros((n_steps, 2))

sim_copy = sim.copy()
for i, t in enumerate(times):
    sim_copy.integrate(t)
    p0, p1, p2 = sim_copy.particles[0], sim_copy.particles[1], sim_copy.particles[2]
    positions_0[i] = (p0.x, p0.y)
    positions_1[i] = (p1.x, p1.y)
    positions_2[i] = (p2.x, p2.y)

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Orbits + Animated Markers")

# Plot full orbits (static)
ax.plot(positions_1[:, 0], positions_1[:, 1], 'b-', alpha=0.5, label="Orbit 1")
ax.plot(positions_2[:, 0], positions_2[:, 1], 'r-', alpha=0.5, label="Orbit 2")
ax.plot(positions_0[:, 0], positions_0[:, 1], 'y-', alpha=0.5, label="Central Mass Path")

# Moving markers
marker_0, = ax.plot(positions_0[0, 0], positions_0[0, 1], 'yo', markersize=8, label="Central Mass")
marker_1, = ax.plot(positions_1[0, 0], positions_1[0, 1], 'bo', markersize=5, label="Object 1")
marker_2, = ax.plot(positions_2[0, 0], positions_2[0, 1], 'ro', markersize=5, label="Object 2")

ax.legend()

# === Animation ===
paused = [False]
frame_idx = [0]

def init():
    frame_idx[0] = 0
    marker_0.set_data([positions_0[0, 0]], [positions_0[0, 1]])
    marker_1.set_data([positions_1[0, 0]], [positions_1[0, 1]])
    marker_2.set_data([positions_2[0, 0]], [positions_2[0, 1]])
    return marker_0, marker_1, marker_2

def update(_):
    if paused[0]:
        return marker_0, marker_1, marker_2

    i = frame_idx[0] % len(times)

    marker_0.set_data([positions_0[i, 0]], [positions_0[i, 1]])
    marker_1.set_data([positions_1[i, 0]], [positions_1[i, 1]])
    marker_2.set_data([positions_2[i, 0]], [positions_2[i, 1]])

    frame_idx[0] += 1
    return marker_0, marker_1, marker_2


ani = FuncAnimation(
    fig, update, init_func=init, interval=30, blit=False,
    save_count=len(times)  # Optional: suppresses warning
)


# === Play/Pause Button ===
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, 'Play/Pause')

def toggle(event):
    paused[0] = not paused[0]

button.on_clicked(toggle)

plt.show()
