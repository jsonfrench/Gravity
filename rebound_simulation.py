import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider, TextBox

import rebound
import matplotlib.pyplot as plt

rebound.horizons.SSL_CONTEXT = 'unverified'

sim = rebound.Simulation()

sim.move_to_com()

sim.integrator = "ias15"    # select an integration scheme for the solver

sim.units = ("kg", "m", "s")    # define the units for the simulation

sim.add(m=1.989e30)
sim.add(m=5.972e24, a=1.5e11, e=0.0167)
sim.add(m=6.39e23, a=2.28e11, e=0.0934)

# sim.add(m=10)
# sim.add(m=1, a=2, e=0.01)
# sim.add(m=1, a=3, e=0.001)

print(f"particle 0 mass: {sim.particles[0].m}")


start = 0
stop = 365 * 3 * (60 * 60 * 24)  # length of time the simulatio runs for (seconds)
steps = stop // (60 * 60 * 24)  # number of steps the simulation takes
delta_time = stop / steps   # length of time step (seconds)
time_space = np.linspace(start, stop, steps)
speed = 1 # amount of time steps to jump forward at a time in the plot

sx, sy, svx, svy, sax, say = [], [], [], [], [], [] 
ex, ey, evx, evy, eax, eay = [], [], [], [], [], [] 
mx, my, mvx, mvy, m_ax, may = [], [], [], [], [], [] 
evx_approx, evy_approx, eax_approx, eay_approx = [], [], [], [] 

for t in time_space:
        sim.integrate(t)

        sx.append(sim.particles[0].x)
        sy.append(sim.particles[0].y)
        ex.append(sim.particles[1].x)
        ey.append(sim.particles[1].y)
        mx.append(sim.particles[2].x)
        my.append(sim.particles[2].y)

        svx.append(sim.particles[0].vx)
        svy.append(sim.particles[0].vy)
        evx.append(sim.particles[1].vx)
        evy.append(sim.particles[1].vy)
        mvx.append(sim.particles[2].vx)
        mvy.append(sim.particles[2].vy)

        sax.append(sim.particles[0].ax)
        say.append(sim.particles[0].ay)
        eax.append(sim.particles[1].ax)
        eay.append(sim.particles[1].ay)
        m_ax.append(sim.particles[2].ax)
        may.append(sim.particles[2].ay)

for i in range(1, len(ex)):
     evx_approx.append((ex[i]-ex[i-1])/delta_time)       # Approximate velocity using position
     evy_approx.append((ey[i]-ey[i-1])/delta_time) 

for i in range(1, len(evx_approx)):
     eax_approx.append((evx_approx[i]-evx_approx[i-1])/delta_time)       # Approximate acceleration using approximated velocities
     eay_approx.append((evy_approx[i]-evy_approx[i-1])/delta_time)  

for i in range(10):
        print(f"{i} Actual vx: {evx[i+1]} Approx vx: {evx_approx[i]}")  # Compare simulation velocities to approximated velocities

for i in range(10):
        print(f"{i} Actual ax: {eax[i+1]} Approx ax: {eax_approx[i]}")  # Compare simulation accelerations to approximated accelerations

# Constants
G = 6.67430e-11 # gravitiaional constant in m^3 kg^-1 s^-2

# Two body system parameters
M = 1.989e30  # mass of central body (Sun) 1.989e30 kg
m1 = 5.972e24 # mass of body 1 (Earth) 5.972e24 kg
m2 = 6.39e23 # mass of body 2 (Mars) 6.39e23 kg

# Convert simulation data lists to np.arrays 
x0s, y0s, vx0s, vy0s = np.array(sx), np.array(sy), np.array(svx), np.array(svy) # Two body Earth positions and velocities
x1s, y1s, vx1s, vy1s = np.array(ex), np.array(ey), np.array(evx), np.array(evy) # Two body Earth positions and velocities
x2s, y2s, vx2s, vy2s = np.array(mx), np.array(my), np.array(mvx), np.array(mvy) # Two body Mars positions and velocities
ax1, ay1, ax2, ay2 = np.array(eax), np.array(eay), np.array(m_ax), np.array(may) # Two body Earth acceleration and mars acceleration
evx_approx, evy_approx = np.array(evx_approx), np.array(evy_approx)
eax_approx, eay_approx = np.array(eax_approx), np.array(eay_approx)
r1s = np.sqrt(x1s**2 + y1s**2) # distances between Earth and Sun
r2s = np.sqrt(x2s**2 + y2s**2) # distances between Mars and Sun
ds= np.sqrt((x2s - x1s)**2 + (y2s - y1s)**2) # distances between Earth and Mars

def mag(x, y):
     return np.sqrt(x**2, y**2)

# Acceleration data for Earth
Ax_mars, Ay_mars = (G * m2 * (x2s - x1s) / ds**3), (G * m2 * (y2s - y1s) / ds**3) # Acceleration of Earth due to Mars (two body simulation)
A_mars_mag = np.sqrt(Ax_mars**2 + Ay_mars**2)

Ax_sun, Ay_sun = -G * M * x1s / r1s**3, -G * M * y1s / r1s**3 # Acceleration of Earth due to the sun (two body simulation)
A_sun_mag = np.sqrt(Ax_sun**2 + Ay_sun**2)

Ax_net, Ay_net = ax1, ay1 # net_simulated acceleration of Earth (two body simulation)
A_net_mag = np.sqrt(Ax_net**2 + Ay_net**2)

Ax_net_comp, Ay_net_comp = Ax_mars + Ax_sun, Ay_mars + Ay_sun # Net acceleration on Earth (Computed using position information) 
A_net_comp_mag = np.sqrt(Ax_net_comp**2 + Ay_net_comp**2) # These should match with the accelerations the simulation returns

Ax_net_approx, Ay_net_approx = eax_approx, eay_approx # Net acceleration on Earth (Approximated using position information) 
A_net_approx_mag = np.sqrt(Ax_net_approx**2 + Ay_net_approx**2) # These should match with the accelerations the simulation returns

Ax_mars_approx, Ay_mars_approx = Ax_net_approx - Ax_sun[:len(Ax_net_approx)], Ay_net_approx - Ay_sun[:len(Ay_net_approx)] # Net acceleration on Earth (Approximated using position information) 
A_mars_approx_mag = np.sqrt(Ax_mars_approx**2 + Ay_mars_approx**2) # These should match with the accelerations the simulation returns

Ax_mars_theoretical, Ay_mars_theoretical = Ax_net - Ax_sun, Ay_net - Ay_sun # Hypothesized acceleration of mars given by Fnet equation
A_mars_theoretical_mag = np.sqrt(Ax_mars_theoretical**2 + Ay_mars_theoretical**2) # Fnet = Fsun + Fmars -> Fnet - Fsun = Fmars

Ax_mars_derived, Ay_mars_derived = Ax_net_comp - Ax_sun, Ay_net_comp - Ay_sun # Hypothesized acceleration of mars given by Fnet equation
A_mars_derived_mag = np.sqrt(Ax_mars_derived**2 + Ay_mars_derived**2) # Fnet = Fsun + Fmars -> Fnet - Fsun = Fmars

# for x in range(100):
#         print(f"{x}. A_net vs A_net_comp {Ax_net[x] - Ax_net_comp[x]}")

def plot_everything():

    # ================ INTERACTIVE PLOT ================ #

    # ==== Create figure and 3 vertically stacked plots ====
    fig, ax_orbit = plt.subplots(1, 1, figsize=(8, 10))
    plt.subplots_adjust(bottom=0.25, hspace=0.4)

    # === Top: Orbit Plot ===
    sun_orbit,   = ax_orbit.plot(x0s, y0s, label='Sun Orbit', alpha=0.5, linewidth=8, color="yellow")
    earth_orbit, = ax_orbit.plot(x1s, y1s, label='Earth Orbit', alpha=0.5, linewidth=8, color="blue")
    mars_orbit,  = ax_orbit.plot(x2s, y2s, label='Mars Orbit', alpha=0.5, linewidth=8, color="red")
    earth_marker, = ax_orbit.plot(x1s[0], y1s[0], 'bo', markersize=8, label='Earth')
    mars_marker, = ax_orbit.plot(x2s[0], y2s[0], 'ro', markersize=8, label='Mars')
    sun_marker, = ax_orbit.plot(x0s[0], y0s[0], 'yo', markersize=8, label='Sun')

    max_range = max(np.max(np.abs(x1s)), np.max(np.abs(x2s)))

    F_mars_vector, = ax_orbit.plot([x1s[0],x1s[0]+Ax_mars[0]/A_mars_mag[0]*max_range],[y1s[0], y1s[0]+Ay_mars[0]/A_mars_mag[0]*max_range], color="red")
    F_sun_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_sun[0]/A_sun_mag[0]*max_range], [y1s[0], y1s[0]+Ay_sun[0]/A_sun_mag[0]*max_range],  color = "yellow")
    # F_net_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_net[0]/A_net_mag[0]*max_range], [y1s[0], y1s[0]+Ay_net[0]/A_net_mag[0]*max_range],  color = "orange")
    # F_net_approx_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_net_approx[0]/A_net_approx_mag[0]*max_range], [y1s[0], y1s[0]+Ay_net_approx[0]/A_net_approx_mag[0]*max_range],  color = "pink")
    # F_mars_approx_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_mars_approx[0]/A_mars_approx_mag[0]*max_range], [y1s[0], y1s[0]+Ay_mars_approx[0]/A_mars_approx_mag[0]*max_range],  color = "green")
#     F_net_comp_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_net_comp[0]/A_net_comp_mag[0]*max_range], [y1s[0], y1s[0]+Ay_net_comp[0]/A_net_comp_mag[0]*max_range],  color = "blue", linewidth=4, alpha=0.5)
    F_mars_theoretical_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_mars_theoretical[0]/A_mars_theoretical_mag[0]*max_range], [y1s[0], y1s[0]+Ay_mars_theoretical[0]/A_mars_theoretical_mag[0]*max_range],  color = "lime")
#     F_mars_derived_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_mars_derived[0]/A_mars_derived_mag[0]*max_range], [y1s[0], y1s[0]+Ay_mars_derived[0]/A_mars_derived_mag[0]*max_range],  color = "purple")

#     max_range = max(np.max(np.abs(x1s)), np.max(np.abs(x2s))) * 200
#     F_mars_vector, = ax_orbit.plot([x1s[0],x1s[0]+Ax_mars[0]*max_range],[y1s[0], y1s[0]+Ay_mars[0]*max_range], color="red")
#     F_sun_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_sun[0]*max_range], [y1s[0], y1s[0]+Ay_sun[0]*max_range],  color = "yellow")
#     F_net_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_net[0]*max_range], [y1s[0], y1s[0]+Ay_net[0]*max_range],  color = "orange")
#     F_net_comp_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_net_comp[0]*max_range], [y1s[0], y1s[0]+Ay_net_comp[0]*max_range],  color = "blue")
#     F_mars_theoretical_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Ax_mars_theoretical[0]*max_range], [y1s[0], y1s[0]+Ay_mars_theoretical[0]*max_range],  color = "lime")
#     max_range = max(np.max(np.abs(x1s)), np.max(np.abs(x2s)))

    ax_orbit.set_aspect('equal')
    ax_orbit.set_xlim(-1.2 * max_range, 1.2 * max_range)
    ax_orbit.set_ylim(-1.2 * max_range, 1.2 * max_range)
    ax_orbit.set_xlabel("x position (m)")
    ax_orbit.set_ylabel("y position (m)")
    ax_orbit.set_title(f"Fnet: {A_net_mag[0]}, Fsun: {A_sun_mag[0]}")
    ax_orbit.grid(True)
    #ax_orbit.legend()

    # === Slider and TextBox ===
    slider_ax = plt.axes([0.2, 0.12, 0.6, 0.03])

    time_slider = Slider(slider_ax, f'Time index (0-{len(time_space)-1})', 0, len(time_space)-1, valinit=0, valstep=1)

    text_ax = plt.axes([0.83, 0.12, 0.1, 0.03])
    time_text = TextBox(text_ax, '', initial="0.00")

    # === Update Function ===

    def update(val):
        idx = int(val)

        # Update orbit markers
        earth_marker.set_data([x1s[idx]], [y1s[idx]])
        mars_marker.set_data([x2s[idx]], [y2s[idx]])
        sun_marker.set_data([x0s[idx]], [y0s[idx]])

        F_mars_vector.set_data([x1s[idx],x1s[idx]+Ax_mars[idx]/A_mars_mag[idx]*max_range],[y1s[idx], y1s[idx]+Ay_mars[idx]/A_mars_mag[idx]*max_range])
        F_sun_vector.set_data([x1s[idx],x1s[idx]+Ax_sun[idx]/A_sun_mag[idx]*max_range], [y1s[idx], y1s[idx]+Ay_sun[idx]/A_sun_mag[idx]*max_range])
        # F_net_vector.set_data([x1s[idx],x1s[idx]+Ax_net[idx]/A_net_mag[idx]*max_range], [y1s[idx], y1s[idx]+Ay_net[idx]/A_net_mag[idx]*max_range])
        # F_net_approx_vector.set_data([x1s[idx],x1s[idx]+Ax_net_approx[idx]/A_net_approx_mag[idx]*max_range], [y1s[idx], y1s[idx]+Ay_net_approx[idx]/A_net_approx_mag[idx]*max_range])
        # F_mars_approx_vector.set_data([x1s[idx],x1s[idx]+Ax_mars_approx[idx]/A_mars_approx_mag[idx]*max_range], [y1s[idx], y1s[idx]+Ay_mars_approx[idx]/A_mars_approx_mag[idx]*max_range])
        # F_net_comp_vector.set_data([x1s[idx],x1s[idx]+Ax_net_comp[idx]/A_net_comp_mag[idx]*max_range], [y1s[idx], y1s[idx]+Ay_net_comp[idx]/A_net_comp_mag[idx]*max_range])
        F_mars_theoretical_vector.set_data([x1s[idx],x1s[idx]+Ax_mars_theoretical[idx]/A_mars_theoretical_mag[idx]*max_range], [y1s[idx], y1s[idx]+Ay_mars_theoretical[idx]/A_mars_theoretical_mag[idx]*max_range])
        # F_mars_derived_vector.set_data([x1s[idx],x1s[idx]+Ax_mars_derived[idx]/A_mars_derived_mag[idx]*max_range], [y1s[idx], y1s[idx]+Ay_mars_derived[idx]/A_mars_derived_mag[idx]*max_range])

        # print(f"{idx}. Axnet: {Ax_net[idx]}, Axapprox: {Ax_net_approx[idx]}")
        # print(f"{idx}. Aynet: {Ay_net[idx]}, Ayapprox: {Ay_net_approx[idx]}")
        # print(f"{idx}. NAxnet: {Ax_net[idx]/A_net_mag[idx]}, NAxapprox: {Ax_net_approx[idx]/A_net_approx_mag[idx]}")
        # print(f"{idx}. Axnetmag: {A_net_mag[idx]}, Axapproxmag: {A_net_approx_mag[idx]}")

        print(f"position {idx}: {ex[idx]}, {ey[idx]}")
        print(f"position {idx+1}: {ex[idx+1]}, {ey[idx]}")
        print(f"position {idx+2}: {ex[idx+2]}, {ey[idx]}")

        print(f"acceleration {idx}: {eax_approx[idx]}, {eay_approx[idx]}")
        print(f"acceleration {idx+1}: {eax_approx[idx+1]}, {eay_approx[idx]}")
        print(f"acceleration {idx+2}: {eax_approx[idx+2]}, {eay_approx[idx]}")

        print(f"A_sun {idx}: {Ax_sun[idx]}, {Ax_sun[idx]}")
        print(f"A_sun {idx+1}: {Ax_sun[idx+1]}, {Ax_sun[idx+1]}")
        print(f"A_sun {idx+2}: {Ax_sun[idx+2]}, {Ax_sun[idx+2]}")

        # max_range = max(np.max(np.abs(x1s)), np.max(np.abs(x2s))) * 200
        # F_mars_vector.set_data([x1s[idx],x1s[idx]+Ax_mars[idx]*max_range],[y1s[idx], y1s[idx]+Ay_mars[idx]*max_range])
        # F_sun_vector.set_data([x1s[idx],x1s[idx]+Ax_sun[idx]*max_range], [y1s[idx], y1s[idx]+Ay_sun[idx]*max_range])
        # F_net_vector.set_data([x1s[idx],x1s[idx]+Ax_net[idx]*max_range], [y1s[idx], y1s[idx]+Ay_net[idx]*max_range])
        # F_net_comp_vector.set_data([x1s[idx],x1s[idx]+Ax_net_comp[idx]*max_range], [y1s[idx], y1s[idx]+Ay_net_comp[idx]*max_range])
        # F_mars_theoretical_vector.set_data([x1s[idx],x1s[idx]+Ax_mars_theoretical[idx]*max_range], [y1s[idx], y1s[idx]+Ay_mars_theoretical[idx]*max_range])

        ax_orbit.set_title(f"Fnet: {A_net_mag[idx]}, Fsun: {A_sun_mag[idx]}")

        # Update text box
        time_text.set_val(f"{val}")
        # time_text.set_val(idx)

        fig.canvas.draw_idle()

    # === TextBox Submit ===
    def submit_text(text):
        try:
            val = float(text)
            val = max(0, min(val, len(time_space)-1))
            time_slider.set_val(val)  # Triggers update
        except ValueError:
            pass


    ######## ADDING PLAY BUTTON TO SLIDER #########
    play_ax = plt.axes([0.4, 0.05, 0.1, 0.04])
    play_button = Button(play_ax, 'Play', hovercolor='0.975')

    # Play button for animation
    playing = [False]  # Use mutable object so we can modify it inside nested function

    def play(event):
        playing[0] = not playing[0]
        if playing[0]:
            play_button.label.set_text('Pause')
            timer.start()
        else:
            play_button.label.set_text('Play')
            timer.stop()

    play_button.on_clicked(play)

    timer_interval = 50  # ~20 FPS <-- seemingly lowest fps

    def advance_slider():
        current_val = time_slider.val
        new_val = (current_val + 1 * speed)  # number of time steps to jump forward
        if new_val >= len(time_space)-1:
            timer.stop()
            playing[0] = False
            play_button.label.set_text('Play')
        else:
            time_slider.set_val(new_val)

    def toggle_angle(event):
        angle_visible[0] = not angle_visible[0]
        angle_toggle_button.label.set_text('Show Angles' if not angle_visible[0] else 'Hide Angles')
        fig.canvas.draw_idle()

    # Create timer
    timer = fig.canvas.new_timer(interval=timer_interval)
    timer.add_callback(advance_slider)

    # Register callbacks
    time_slider.on_changed(update)
    time_text.on_submit(submit_text)

    toggle_ax = plt.axes([0.52, 0.05, 0.18, 0.04])
    toggle_button = Button(toggle_ax, 'Hide Mars', hovercolor='0.975')
    mars_visible = [True]  # Mutable flag

    angle_toggle_ax = plt.axes([0.72, 0.05, 0.18, 0.04])
    angle_toggle_button = Button(angle_toggle_ax, 'Hide Angles', hovercolor='0.975')
    angle_visible = [True]

    def toggle_mars(event):
        mars_visible[0] = not mars_visible[0]
        mars_marker.set_visible(mars_visible[0])
        mars_orbit.set_visible(mars_visible[0])
        toggle_button.label.set_text('Show Mars' if not mars_visible[0] else 'Hide Mars')
        fig.canvas.draw_idle()

    fov = 5.0
    last_angle = 0.0

    toggle_button.on_clicked(toggle_mars)
    angle_toggle_button.on_clicked(toggle_angle)

    ax_orbit.legend()
    plt.show()

    # Initialize plot
    update(0)

plot_everything()

