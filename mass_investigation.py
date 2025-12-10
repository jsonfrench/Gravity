import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox
from scipy.signal import find_peaks
from scipy.stats import zscore
from matplotlib.patches import Wedge

import sys

# === Setup Simulation ===
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

# G = 6.67430e-11
G = sim.G

# Contrived Sun–Earth–Mars system
central_mass = 300000.0
r1, r2 = 1.0, 2 
m1 = 1.0
m2 = 0.107 
v1 = np.sqrt(G * central_mass / r1)
v2 = np.sqrt(G * central_mass / r2) * 1.1

# # === base example
# sim.add(m=central_mass)   # Sun
# sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
# sim.add(m=m2, x=r2, y=0, vy=v2)  # Mars
# t_max = 2e4

# === elliptical inside
# sim.add(m=central_mass)   # Sun
# sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
# sim.add(m=1, a=2, e=0.7) 
# t_max = 3e4

# # === Real world example
# rebound.horizons.SSL_CONTEXT = "unverified"
# sim.add("Sun")   # Sun
# sim.add("Earth")  # Earth
# sim.add("Mars")  # Mars
# t_max = 60*60*24*365.25*20


# === Time Setup ===
n_steps = int(1e4)
t_max = n_steps
times = np.linspace(0, t_max, n_steps)
# dt = times[1] - times[0]
dt = 1 

WEEK = 60*60*24*7

n_steps = WEEK * 52
min_mass = 0.01
max_mass = 1000
n_masses = 2
masses = np.linspace(min_mass, max_mass, n_masses)


# === Integrate and Record Positions ===
positions_0 = np.zeros((n_masses, 2))
positions_1 = np.zeros((n_masses, 2))
positions_2 = np.zeros((n_masses, 2))

accelerations_1 = np.zeros((n_masses, 2))
accelerations_2 = np.zeros((n_masses, 2))

for i, m in enumerate(masses):
    sim = rebound.Simulation()
    sim.units = ('s','m','kg')
    sim.integrator = "ias15"

    sim.add(m=central_mass)
    sim.add(m=m1, x=r1, y=0, vy=v1)
    sim.add(m=m,  x=r2, y=0, vy=v2)

    sim.integrate(n_steps)

print(positions_1)

# === Plot Setup ===
# fig, (ax_orbit, ax_combined, ax_delta) = plt.subplots(3, 1, figsize=(6, 9))
fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(bottom=0.25)

ax.set_aspect("equal")
ax.grid(True)

marker_sun,   = ax.plot([], [], 'yo', markersize=8)
marker_earth, = ax.plot([], [], 'bo', markersize=5)
marker_mars,  = ax.plot([], [], 'ro', markersize=5)

def update(idx):
    idx = int(idx)
    marker_sun.set_data([positions_0[idx,0]], [positions_0[idx,1]])
    marker_earth.set_data([positions_1[idx,0]], [positions_1[idx,1]])
    marker_mars.set_data([positions_2[idx,0]], [positions_2[idx,1]])
    fig.canvas.draw_idle()

ax_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
slider = Slider(ax_slider, 'Mass Index', 0, n_masses - 1, valinit=0, valstep=1)

slider.on_changed(update)

update(0)
plt.show()
