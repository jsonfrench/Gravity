import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox
from scipy.signal import find_peaks
from matplotlib.patches import Wedge
import sys

# === Setup Simulation ===
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

G = 6.67430e-11

# Contrived Sun–Earth–Mars system
central_mass = 300000.0
r1, r2 = 1.0, 2
m1 = 1.0
m2 = 0.107
v1 = np.sqrt(G * central_mass / r1)
v2 = np.sqrt(G * central_mass / r2)

# === base example
sim.add(m=central_mass)   # Sun
sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
sim.add(m=m2, x=r2, y=0, vy=v2)  # Mars
t_max = 2e4

# # === elliptical inside
# sim.add(m=central_mass)   # Sun
# sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
# sim.add(m=1, a=2, e=0.7) 
# t_max = 3e4

# === Time Setup ===
# t_max = 2e4
n_steps = int(1e4)
times = np.linspace(0, t_max, n_steps)
dt = times[1] - times[0]

# === Integrate and Record Positions ===
positions_0 = np.zeros((n_steps, 2))
positions_1 = np.zeros((n_steps, 2))
positions_2 = np.zeros((n_steps, 2))

sim_copy = sim.copy()
for i, t in enumerate(times):
    sim_copy.integrate(t)
    p0, p1, p2 = sim_copy.particles
    positions_0[i] = [p0.x, p0.y]
    positions_1[i] = [p1.x, p1.y]
    positions_2[i] = [p2.x, p2.y]

# === Compute Accelerations and Velocities ===
accel_earth_approx = np.zeros_like(positions_1)
for i in range(2, n_steps):
    accel_earth_approx[i - 1] = (positions_1[i] - 2 * positions_1[i - 1] + positions_1[i - 2]) / dt**2

accel_sun = np.zeros_like(positions_1)
for i in range(n_steps):
    r_vec = positions_1[i] - positions_0[i]
    r = np.linalg.norm(r_vec)
    if r > 0:
        accel_sun[i] = -G * central_mass * r_vec / r**3

# Perturbation acceleration
accel_mars = accel_earth_approx - accel_sun

# === Plot Setup ===
fig, ax_orbit = plt.subplots(1, 1, figsize=(12, 9))
plt.subplots_adjust(bottom=0.25, hspace=0.35)
idx = 0 # Animation starts on this frame

# 1️⃣ Orbit Plot
# ax_orbit.set_xlim(-1.6, 1.6)
# ax_orbit.set_ylim(-1.6, 1.6)
ax_orbit.set_aspect('equal')
ax_orbit.set_title("Orbital Motion with Normalized Perturbation and Earth–Mars Vectors")
ax_orbit.grid(True)

ax_orbit.plot(positions_1[:, 0], positions_1[:, 1], 'b-', alpha=0.3)
ax_orbit.plot(positions_2[:, 0], positions_2[:, 1], 'r-', alpha=0.3)
ax_orbit.plot(positions_0[:, 0], positions_0[:, 1], color='gold', lw=0.5, alpha=0.3)

marker_sun, = ax_orbit.plot([], [], 'yo', markersize=8)
marker_earth, = ax_orbit.plot([], [], 'bo', markersize=5)
marker_mars, = ax_orbit.plot([], [], 'ro', markersize=5)
pert_arrow = None
mars_arrow = None
ax_orbit.legend(loc="upper right")

flashlight = Wedge((positions_1[1]),
                    positions_1[0,0]*0.5, 
                    -2*np.pi*positions_1[0,0]*0.5*180/np.pi/16, 
                    2*np.pi*positions_1[0,0]*0.5*180/np.pi/16, 
                    color = "gold", alpha = 0.5)
ax_orbit.add_patch(flashlight)

# === Update Function ===
def update(i):
    global pert_arrow, mars_arrow
    marker_sun.set_data([positions_0[i, 0]], [positions_0[i, 1]])
    marker_earth.set_data([positions_1[i, 0]], [positions_1[i, 1]])
    marker_mars.set_data([positions_2[i, 0]], [positions_2[i, 1]])
    flashlight.set_center(positions_1[i])
    if is_visible(
        positions_1[i, 0], positions_1[i,1],
        positions_2[i, 0], positions_2[i, 1], 
        angle.val, 
        distance.val, 
        field_of_view
    ): 
        marker_mars.set_color("green")
    else: 
        marker_mars.set_color("red")

    for arrow in [pert_arrow, mars_arrow]:
        if arrow is not None:
            arrow.remove()
    pert_arrow = mars_arrow = None

    # Normalized arrows from Earth
    earth_pos = positions_1[i]
    pert_vec = accel_mars[i]
    mars_vec = positions_2[i] - positions_1[i]
    if np.linalg.norm(pert_vec) > 0 and np.linalg.norm(mars_vec) > 0:
        pert_unit = pert_vec / np.linalg.norm(pert_vec)
        mars_unit = mars_vec / np.linalg.norm(mars_vec)
        arrow_len = 0.3
        pert_arrow = ax_orbit.arrow(earth_pos[0], earth_pos[1],
                                    pert_unit[0]*arrow_len, pert_unit[1]*arrow_len,
                                    color='lime', head_width=0.03)
        mars_arrow = ax_orbit.arrow(earth_pos[0], earth_pos[1],
                                    mars_unit[0]*arrow_len, mars_unit[1]*arrow_len,
                                    color='red', head_width=0.03)

# === Telescope Controls === 
ax_angle        = plt.axes([0.1, 0.55, 0.15, 0.03])
ax_distance     = plt.axes([0.1, 0.50, 0.15, 0.03])
ax_resolution   = plt.axes([0.1, 0.45, 0.15, 0.03])
angle       = Slider(ax_angle, "Angle", 0, 2*np.pi, valinit = 0)
distance    = Slider(ax_distance, "Distance", 0, max(np.amax(positions_1), np.amax(positions_2)) * 2 * np.sqrt(2), valinit = positions_1[0,0]*0.5)
resolution  = Slider(ax_resolution, "Resolution", 0, 1, valinit=2*np.pi*distance.val/16)
resolution_val = resolution.val # workaround because adjusting slider bounds sucks
field_of_view = resolution_val / distance.val
ax_angle.set_title("Telescope Controls")

def update_angle(val):
    flashlight.set_theta1((angle.val*180/np.pi)-(field_of_view*180/np.pi/2))
    flashlight.set_theta2((angle.val*180/np.pi)+(field_of_view*180/np.pi/2))
def update_distance(val):
    flashlight.set_radius(distance.val)
    global field_of_view
    global resolution_val
    resolution_val = min(resolution_val, 2*np.pi*distance.val)
    resolution.set_val(resolution_val/(2*np.pi*distance.val)) 
    field_of_view = resolution_val / distance.val 
    flashlight.set_theta1((angle.val*180/np.pi)-(field_of_view*180/np.pi/2))
    flashlight.set_theta2((angle.val*180/np.pi)+(field_of_view*180/np.pi/2))
def update_resolution(val):
    global field_of_view
    global resolution_val
    resolution_val = resolution.val*2*np.pi*distance.val
    field_of_view = resolution_val / distance.val 
    flashlight.set_theta1((angle.val*180/np.pi)-(field_of_view*180/np.pi/2))
    flashlight.set_theta2((angle.val*180/np.pi)+(field_of_view*180/np.pi/2))    

angle.on_changed(update_angle)
distance.on_changed(update_distance)
resolution.on_changed(update_resolution)

def is_visible(x0, y0, x1, y1, angle, distance, fov):
    angle_between = np.arctan2(y0-y1, x0-x1)+np.pi

    is_within_range = np.sqrt((x1-x0)**2 + (y1-y0)**2) < distance
    is_within_fov = min(np.abs(angle-angle_between),(2*np.pi)-np.abs(angle-angle_between)) < fov/2 # workaround, should use dot product
    is_within_sector = is_within_range and is_within_fov

    return is_within_sector

# === Time Slider and Button ===
ax_time_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
time_slider = Slider(ax_time_slider, 'Time', 0, n_steps - 1, valinit=n_steps-1, valstep=1)

paused = False
def toggle(event):
    global paused
    paused = not paused

ax_button = plt.axes([0.86, 0.11, 0.1, 0.04])
button = Button(ax_button, 'Play/Pause')
button.on_clicked(toggle)

def time_slider_update(val):
    global idx
    idx = int(time_slider.val)
    update(idx)
time_slider.on_changed(time_slider_update)

# === Textbox for Jump-to-Frame ===
axbox = plt.axes([0.1, 0.01, 0.2, 0.075])
text_box = TextBox(axbox, 'Jump to Frame:', initial="0")

def jump_to_frame(text):
    try:
        global idx
        idx = int(text)
        update(idx)         # update the plots to the current time value
        time_slider.set_val(idx) # update the time slider to be on the current time value
    except:
        print("Invalid frame index")

text_box.on_submit(jump_to_frame)


def animate(frame):
    global idx
    if not paused:
        idx = (idx + 1) % n_steps   # Increment counter and wrap around at the end 
        time_slider.set_val(idx)
    return []

ani = FuncAnimation(fig, animate, frames=times, interval=10, blit=True, repeat=True)

plt.show()
