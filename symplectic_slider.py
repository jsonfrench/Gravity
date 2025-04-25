import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

from matplotlib import pyplot as plt

import desolver as de
import numpy as np
from constants import G

def Fij(ri, rj, G):
    rel_r = rj - ri
    return G*(1/np.linalg.norm(rel_r, ord=2)**3)*rel_r

def rhs(t, state, masses, G):
    total_acc = np.zeros_like(state)

    for idx, (ri, mi) in enumerate(zip(state, masses)):
        for jdx, (rj, mj) in enumerate(zip(state[idx+1:], masses[idx+1:])):
            partial_force = Fij(ri[:3], rj[:3], G)
            total_acc[idx, 3:]       += partial_force * mj
            total_acc[idx+jdx+1, 3:] -= partial_force * mi

    total_acc[:, :3] = state[:, 3:]

    return total_acc

# Msun = 1.98847*10**30   ## Mass of the Sun, kg
Msun = 331658
# AU   = 149597871e3      ## 1 Astronomical Unit, m
AU   = 1      ## 1 Astronomical Unit, m
# year = 365.25*24*3600   ## 1 year, s
# G    = 4*np.pi**2       ## in solar masses, AU, years
V    = np.sqrt(G)        ## Speed scale corresponding to the orbital speed required for a circular orbit at 1AU with a period of 1yr

vy1 = np.sqrt(G*Msun/14) 
vy2 = np.sqrt(G*Msun/15) 

initial_state = np.array([
  # [x,     y,                  z,    vx,                    vy,     vs ]
    [0.0,   0.0,                0.0,  0.0,                  0.0,    0.0],   # sun
    [14.0,   0.0,                0.0,  0.0,                  vy1,    0.0],   # body 1
    [15,   0.0,                0.0,  0.0,                   vy2,    0.0], # body 2
], dtype=np.longdouble)

masses = np.array([
    Msun,
    1,
    1
], dtype=np.longdouble)

rhs(0.0, initial_state, masses, G)


t_max = 400000.0

a = de.OdeSystem(rhs, y0=initial_state, dense_output=True, t=(0, t_max), dt=0.001, rtol=de.backend.tol_epsilon(initial_state.dtype), atol=de.backend.tol_epsilon(initial_state.dtype), constants=dict(G=G, masses=masses))
a.method = "RK1412"

a.integrate()

com_motion = np.sum(a.y[:, :, :] * masses[None, :, None], axis=1) / np.sum(masses)

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,3,(1,3))
# ax2 = fig.add_subplot(132, aspect=1)
# ax3 = fig.add_subplot(133, aspect=1)

ax1.set_xlabel("x (AU)")
ax1.set_ylabel("y (AU)")
# ax2.set_xlabel("y (AU)")
# ax2.set_ylabel("z (AU)")
# ax3.set_xlabel("z (AU)")
# ax3.set_ylabel("x (AU)")

print(a.y.shape[0])

for i in range(a.y.shape[1]):
    ax1.plot(a.y[:, i, 0], a.y[:, i, 1], color=f"C{i}")
    # ax1.plot(np.linspace(1,a.y.shape[0],a.y.shape[0]), a.y[:, i, 1], color=f"C{i}")
    # ax2.plot(a.y[:, i, 1], a.y[:, i, 2], color=f"C{i}")
    # ax3.plot(a.y[:, i, 2], a.y[:, i, 0], color=f"C{i}")

ax1.scatter(com_motion[:, 0], com_motion[:, 1], color='k')
# ax2.scatter(com_motion[:, 1], com_motion[:, 2], color='k')
# ax3.scatter(com_motion[:, 2], com_motion[:, 0], color='k')

plt.tight_layout()
plt.show()

def plot_nbody_orbit_interactive(t, y, planet_colors=None, xlim=(-15, 15), ylim=(-15, 15)):
    num_bodies = y.shape[1]
    num_points = y.shape[0]
    
    if planet_colors is None:
        planet_colors = [f"C{i}" for i in range(num_bodies)]

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Interactive Orbit Simulation")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")

    # Initial index
    # initial_index = 1000
    initial_index = min(1000, num_points - 1)
    orbits = []
    markers = []

    for i in range(num_bodies):
        orbit_line, = ax.plot(y[:initial_index, i, 0], y[:initial_index, i, 1], color=planet_colors[i])
        orbit_dot = ax.scatter(y[initial_index, i, 0], y[initial_index, i, 1], color=planet_colors[i], s=30)
        orbits.append(orbit_line)
        markers.append(orbit_dot)

    # Slider axis and widget
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    index_slider = Slider(
        ax=ax_slider,
        label="Time index",
        valmin=0,
        valmax=num_points - 1,
        valinit=initial_index,
        valstep=1,
    )

    # Update function
    def update(val):
        idx = int(index_slider.val)
        for i in range(num_bodies):
            orbits[i].set_xdata(y[:idx, i, 0])
            orbits[i].set_ydata(y[:idx, i, 1])
            markers[i].set_offsets([y[idx, i, 0], y[idx, i, 1]])
        fig.canvas.draw_idle()

    index_slider.on_changed(update)

    # Reset button
    resetax = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    button.on_clicked(lambda event: index_slider.reset())

    plt.show()

plot_nbody_orbit_interactive(a.t, a.y)
