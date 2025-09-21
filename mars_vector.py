import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider, TextBox
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.signal import find_peaks

# Constants
G = 6.67430e-11 # gravitiaional constant in m^3 kg^-1 s^-2

# Sympletic integrator coefficents (Yoshida 4th order)
# note: the coefficients control the integration steps to preserve physical properties
w0 = -np.power(2, 1/3) / (2 - np.power(2,1/3))
w1 = 1 / (2 - np.power(2, 1/3))
c = [w1 / 2, (w0 + w1) / 2, (w0 + w1) / 2, w1 / 2] # position update weights
d = [w1, w0, w1] # velocity update weights

# Function to compute accelerations on two orbiting bodies affected by central mass and each other
def compute_acceleration_two_orbiting_bodies(x1,y1,x2,y2,M,m1,m2):

    r1 = np.sqrt(x1**2 + y1**2) # distance between m1 and M
    r2 = np.sqrt(x2**2 + y2**2) # distance between m2 and M
    d= np.sqrt((x2 - x1)**2 + (y2 - y1)**2) # distance between m1 and m2

    # Compute acceleration on mass 1 (e.g., Earth):
    # First term: attraction to central mass (Sun)
    # Second term: gravitational pull from mass 2 (e.g., Mars)
    ax1 = -G * M * x1 / r1**3 + (G * m2 * (x2 - x1) / d**3) # acceleration in x direction on mass 1
    ay1 = -G * M * y1 / r1**3 + (G * m2 * (y2 - y1) / d**3) # acceleration in y direction on mass 1

    # Compute acceleration on mass 2 (e.g., Mars):
    ax2 = -G * M * x2 / r2**3 + (G * m1 * (x1 - x2) / d**3) # acceleration in x direction on mass 2
    ay2 = -G * M * y2 / r2**3 + (G * m1 * (y1 - y2) / d**3) # acceleration in y direction on mass 2

    return ax1, ay1, ax2, ay2

# Function to compute acceleration on a single orbiting body affected only by the central mass
def compute_acceleration_one_orbiting_body(x1,y1,M,m):

    r = np.sqrt(x1**2 + y1**2) # distance between m1 and M

    ax = -G * M * x1 / r**3 # acceleration in x direction on mass 1
    ay = -G * M * y1 / r**3  # acceleration in y direction on mass 1

    return ax, ay

# Symplectic integrator for two orbiting bodies under mutual and central gravitational forces
def symplectic_integrate_two_body(IVP, dt, steps, M, m1, m2, delay_step):

    # Unpack initial conditions: positions and velocities of both bodies
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP # initial value conditions

    solution = np.zeros((steps,8)) # initialze array of solutions, this should be the size of the intial conditons by the number of time steps
    solution[0] = IVP # the first time step soltuion is set to the initial values
    accelerations = np.zeros((steps,4))
    accelerations[0] = [0,0,0,0]

    for i in range (1,steps):
        # First partial position update with coefficient c[0]
        x1 += c[0] * dt * vx1
        y1 += c[0] * dt * vy1
        x2 += c[0] * dt * vx2
        y2 += c[0] * dt * vy2

        # Loop through the three substeps of the 4th order symplectic integration
        for j in range(3): #  there are 3 substeps for 4th order integration

            if i >= delay_step:
                # Compute accelerations on both bodies due to gravitational forces
                ax1, ay1, ax2, ay2 = compute_acceleration_two_orbiting_bodies(x1, y1, x2, y2, M, m1, m2)

            else:
                ax1, ay1 = compute_acceleration_one_orbiting_body(x1, y1, M , m1)
                ax2, ay2 = compute_acceleration_one_orbiting_body(x2, y2, M, m2)

            # Update velocities using the computed accelerations weighted by d[j]
            vx1 += d[j] * dt * ax1
            vy1 += d[j] * dt * ay1
            vx2 += d[j] * dt * ax2
            vy2 += d[j] * dt * ay2

            # Update positions again with updated velocities weighted by c[j+1]
            x1 += c[j + 1] * dt * vx1
            y1 += c[j + 1] * dt * vy1
            x2 += c[j + 1] * dt * vx2
            y2 += c[j + 1] * dt * vy2

       # Save the positions and velocities at this timestep
        solution[i] = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        accelerations[i] = [ax1,ay1,ax2,ay2]

    return solution, accelerations

# Symplectic integrator for a single orbiting body around a central mass
def symplectic_integrate_one_body(IVP, dt, steps, M, m):

    # Unpack initial position and velocity
    x, y, vx, vy = IVP # initial value conditions

    solution = np.zeros((steps,4)) # initialze array of solutions, this should be the size of the intial conditons by the number of time steps
    solution[0] = IVP # the first time step soltuion is set to the initial values
    accelerations = np.zeros((steps,2))
    accelerations[0] = [0,0]

    for i in range (1,steps):

        # initial position update with c[0], first partial position update
        x += c[0] * dt * vx
        y += c[0] * dt * vy

        # Perform the three substeps of the integrator
        for j in range(3): #  there are 3 substeps for 4th order integration

             # Calculate acceleration due to central mass gravity
            ax, ay = compute_acceleration_one_orbiting_body(x, y,  M, m)

            # Update velocities
            vx += d[j] * dt * ax
            vy += d[j] * dt * ay

            # Update positions
            x += c[j + 1] * dt * vx
            y += c[j + 1] * dt * vy

        # stoe the solution at the current time step
        solution[i] = [x, y, vx, vy]
        accelerations[i] = [ax,ay]

    return solution, accelerations

# Two body system parameters
M = 1.989e30  # mass of central body (Sun) 1.989e30 kg
m1 = 5.972e24 # mass of body 1 (Earth) 5.972e24 kg
m2 = 6.39e23  # mass of body 2 (Mars) 6.39e23 kg
angleE = 0 * (np.pi/180)           # Earth initial angle 
rad1 = 1.5e11                      # Earth initial distance 
x1 = rad1 * np.cos(angleE)         # Earth initial x-position (~1 AU)
y1 = rad1 * np.sin(angleE)         # Earth initial y-position 
r1 = np.sqrt(x1**2 + y1 **2)       # Earth initial distance from origin (redundant) 
velE =  np.sqrt(abs(G * M/ rad1))  # Earth initial velocity 
vx1 = velE * -np.sin(angleE)       # Earth initial velocity in x direction
vy1 = velE * np.cos(angleE)        # Earth initial velocity in y direction
angleM = 52 * (np.pi/180)       # Mars initial angle 
rad2 = 2.28e11                  # Mars initial distance 
x2 = rad2*np.cos(angleM)        # Mars initial x-position 
y2 = rad2*np.sin(angleM)        # Mars initial y-position 
r2 = np.sqrt(x2**2 + y2 **2)    # Mars initial distance from origin (redundant) 
velM = np.sqrt(abs(G * M/rad2)) # Mars initial velocity 
vx2 =velM * -np.sin(angleM)     # Mars initial velocity in x direction
vy2 = velM * np.cos(angleM)     # Mars initial velocity in y direction

T_earth_theoretical = np.sqrt((4*np.pi**2 * r1 **3) / (G * M)) /  3.154e+7 # Earth orbtial period (1 year)
T_mars_theoretical = np.sqrt((4*np.pi**2 * r2 **3) / (G * M)) /  3.154e+7  # Mars orbital period

# Inital conditions for each simulation
IVP_2body = [x1, y1, vx1, vy1, x2, y2, vx2, vy2 ] # Two- body intial conditions
IVP_Earth = [x1, y1, vx1, vy1] # One-body Earth intial conditions
IVP_Mars  = [x2, y2, vx2, vy2 ] # One-body Mars intial conditions

# Time information for the simulation
dt = (60 ** 2)*24  # time step value (duration of each time step in seconds), inital set to 1 day
total_time = 100 # amount of time to run the simulation for (in years)
total_time_seconds = total_time * 31556952 # total simulation time (in seconds)
steps = int(total_time_seconds / dt) # number of time steps to run the simulation for 
orbital_period_earth = 2* np.pi* np.sqrt((rad1**3)/(G * M)) 
delay_time = 0.25 * orbital_period_earth    # Ignore physics for a quarter year (workaround) 
delay_step = int(delay_time/dt) # Number of time steps to ingnore physics for 
t = np.arange(steps) * dt / (60*60*24*365.25)   # years for the x-axis

# Run the simulations
sol_2body, acc_2body = symplectic_integrate_two_body(IVP_2body, dt, steps, M, m1, m2, delay_step)
sol_Earth, acc_1bodyE = symplectic_integrate_one_body(IVP_Earth, dt, steps, M, m1)
sol_Mars, acc_1bodyM = symplectic_integrate_one_body(IVP_Mars, dt, steps, M, m2)

# Extract simulation data for plotting and computing force vectors 
x1s, y1s, vx1s, vy1s = sol_2body[:, 0], sol_2body[:, 1], sol_2body[:, 2], sol_2body[:, 3] # Two body Earth positions and velocities
x2s, y2s, vx2s, vy2s = sol_2body[:, 4], sol_2body[:, 5], sol_2body[:, 6], sol_2body[:, 7] # Two body Mars positions and velocities
ax1, ay1, ax2, ay2 = acc_2body[:,0],acc_2body[:,1],acc_2body[:,2],acc_2body[:,3] # Two body Earth acceleration and mars acceleration
xE, yE, vxE, vyE = sol_Earth[:,0], sol_Earth[:,1], sol_Earth[:,2], sol_Earth[:,3] # One body earth positions and velocities
xM, yM, vxM, vyM = sol_Mars[:,0], sol_Mars[:,1], sol_Mars[:,2], sol_Mars[:,3]  # One body Mars positions and velocities 
axE, ayE = acc_1bodyE[:,0],acc_1bodyE[:,1] # One body Earth accelerations
axM, ayM = acc_1bodyM[:,0],acc_1bodyM[:,1] # One body Mars accelerations
r1s = np.sqrt(x1s**2 + y1s**2) # distances between Earth and Sun
r2s = np.sqrt(x2s**2 + y2s**2) # distances between Mars and Sun
ds= np.sqrt((x2s - x1s)**2 + (y2s - y1s)**2) # distances between Earth and Mars

# Acceleration data for Earth
Ax_mars = (G * m2 * (x2s - x1s) / ds**3) # Acceleration of Earth due to Mars (two body simulation)
Ay_mars = (G * m2 * (y2s - y1s) / ds**3)
A_mars_mag = np.sqrt(Ax_mars**2 + Ay_mars**2)
Ax_mars_norm = Ax_mars / A_mars_mag
Ay_mars_norm = Ay_mars / A_mars_mag

Ax_sun = -G * M * x1s / r1s**3           # Acceleration of Earth due to the sun (two body simulation)
Ay_sun = -G * M * y1s / r1s**3
A_sun_mag = np.sqrt(Ax_sun**2 + Ay_sun**2)
Ax_sun_norm = Ax_sun / A_sun_mag
Ay_sun_norm = Ay_sun / A_sun_mag

Ax_net = Ax_mars + Ax_sun                  # Net acceleration on Earth (Computed using position information) 
Ay_net = Ay_mars + Ay_sun                  # These should match with the accelerations the simulation returns
A_net_mag = np.sqrt(Ax_net**2 + Ay_net**2)
Ax_net_norm = Ax_net / A_net_mag
Ay_net_norm = Ay_net / A_net_mag

Ax_net_simulated = ax1                             # net_simulated acceleration of Earth (two body simulation)
Ay_net_simulated = ay1
A_net_simulated_mag = np.sqrt(Ax_net_simulated**2 + Ay_net_simulated**2)
Ax_net_simulated_norm = Ax_net_simulated / A_net_simulated_mag
Ay_net_simulated_norm = Ay_net_simulated / A_net_simulated_mag

Ax_mars_theoretical =  Ax_net - Ax_sun             # Hypothesized acceleration of mars given by Fnet equation
Ay_mars_theoretical =  Ay_net - Ay_sun             # Fnet = Fsun + Fmars -> Fnet - Fsun = Fmars
A_mars_theoretical_mag = np.sqrt(Ax_mars_theoretical**2 + Ay_mars_theoretical**2)
Ax_mars_theoretical_norm = Ax_mars_theoretical / A_mars_theoretical_mag
Ay_mars_theoretical_norm = Ay_mars_theoretical / A_mars_theoretical_mag

Ax_mars_theoretical_sim =  Ax_net_simulated - Ax_sun             # Hypothesized acceleration of mars given by Fnet equation
Ay_mars_theoretical_sim =  Ay_net_simulated - Ay_sun             # Fnet = Fsun + Fmars -> Fnet - Fsun = Fmars
A_mars_theoretical_sim_mag = np.sqrt(Ax_mars_theoretical_sim**2 + Ay_mars_theoretical_sim**2)
Ax_mars_theoretical_sim_norm = Ax_mars_theoretical_sim / A_mars_theoretical_sim_mag
Ay_mars_theoretical_sim_norm = Ay_mars_theoretical_sim / A_mars_theoretical_sim_mag

# Convert accelerations to forces
Fx_mars = Ax_mars * m1                   # Force of Mars on Earth (two body simulation) 
Fy_mars = Ay_mars * m1 
F_mars_mag = np.sqrt(Fx_mars**2 + Fy_mars**2)
Fx_mars_norm = Fx_mars / F_mars_mag 
Fy_mars_norm = Fy_mars / F_mars_mag 

Fx_sun = Ax_sun * m1                     # Force of sun on Earth (two body simulation)
Fy_sun = Ay_sun * m1
F_sun_mag = np.sqrt(Fx_sun**2 + Fy_sun**2)
Fx_sun_norm = Fx_sun / F_sun_mag 
Fy_sun_norm = Fy_sun / F_sun_mag 

Fx_net = Ax_net * m1                     # Net force on Earth (two body simulation)
Fy_net = Ay_net * m1                     # Computed using position values
F_net_mag = np.sqrt(Fx_net**2 + Fy_net**2)
Fx_net_norm = Fx_net / F_net_mag 
Fy_net_norm = Fy_net / F_net_mag 

Fx_net_simulated = Ax_net_simulated * m1                     # Net force on Earth (two body simulation)
Fy_net_simulated = Ay_net_simulated * m1                     # Computed using simulation acceleration values
F_net_simulated_mag = np.sqrt(Fx_net_simulated**2 + Fy_net_simulated**2)
Fx_net_simulated_norm = Fx_net_simulated / F_net_simulated_mag 
Fy_net_simulated_norm = Fy_net_simulated / F_net_simulated_mag 

Fx_mars_theoretical =  Fx_net - Fx_sun                 # Hypothesized force of earth given Fnet equation 
Fy_mars_theoretical =  Fy_net - Fy_sun                 # Uses two body simulation for F_sun
F_mars_theoretical_mag = np.sqrt(Fx_mars_theoretical**2 + Fy_mars_theoretical**2)
Fx_mars_theoretical_norm =  Fx_mars_theoretical / F_mars_theoretical_mag
Fy_mars_theoretical_norm =  Fy_mars_theoretical / F_mars_theoretical_mag

Fx_mars_theoretical_sim =  Fx_net_simulated - Fx_sun                 # Hypothesized force of earth given Fnet equation 
Fy_mars_theoretical_sim =  Fy_net_simulated - Fy_sun                 # Uses two body simulation for F_sun
F_mars_theoretical_sim_mag = np.sqrt(Fx_mars_theoretical_sim**2 + Fy_mars_theoretical_sim**2)
Fx_mars_theoretical_sim_norm =  Fx_mars_theoretical_sim / F_mars_theoretical_sim_mag
Fy_mars_theoretical_sim_norm =  Fy_mars_theoretical_sim / F_mars_theoretical_sim_mag

# Check calculated accelerations match returned simulation acceleration values
print(f"Ax_net {Ax_net}, Ax_net_simulated {Ax_net_simulated}")

# Check if precision of acceleration values match the precision of force vectors
print(f"Ax_net_norm {Ax_net_norm}, Fx_net_norm {Fx_net_norm}")

def plot_everything():

    # ================ INTERACTIVE PLOT ================ #

    # ==== Create figure and 3 vertically stacked plots ====
    fig, ax_orbit = plt.subplots(1, 1, figsize=(8, 10))
    plt.subplots_adjust(bottom=0.25, hspace=0.4)

    # === Top: Orbit Plot ===
    ax_orbit.plot(x1s, y1s, label='Earth (2-body)', alpha=0.5, linewidth=8)
    mars_orbit_line, = ax_orbit.plot(x2s, y2s, label='Mars (2-body)', alpha=0.5, linewidth=8)
    ax_orbit.plot(0, 0, 'yo', label='Sun')
    earth_marker, = ax_orbit.plot(x1s[0], y1s[0], 'bo', markersize=8, label='Earth')
    mars_marker, = ax_orbit.plot(x2s[0], y2s[0], 'ro', markersize=8, label='Mars')
    angle_guess_line, = ax_orbit.plot([], [], 'm--', label='User Angle Line')

    max_range = max(np.max(np.abs(x1s)), np.max(np.abs(x2s)))

    # F_mars_vector, = ax_orbit.plot([x1s[0],x1s[0]+Fx_mars_theoretical_norm[0]*max_range],[y1s[0], y1s[0]+Fy_mars_theoretical_norm[0]*max_range], color="red")
    F_sun_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Fx_sun_norm[0]*max_range], [y1s[0], y1s[0]+Fy_sun_norm[0]*max_range],  color = "yellow")
    F_net_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Fx_net_norm[0]*max_range], [y1s[0], y1s[0]+Fy_net_norm[0]*max_range],  color = "orange")
    F_mars_theoretical_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Fx_mars_theoretical_norm[0]*max_range], [y1s[0], y1s[0]+Fy_mars_theoretical_norm[0]*max_range],  color = "purple")
    # F_mars_theoretical_sim_vector,  = ax_orbit.plot([x1s[0],x1s[0]+Fx_mars_theoretical_sim_norm[0]*max_range], [y1s[0], y1s[0]+Fy_mars_theoretical_sim_norm[0]*max_range],  color = "lime")

    ax_orbit.set_aspect('equal')
    ax_orbit.set_xlim(-1.2 * max_range, 1.2 * max_range)
    ax_orbit.set_ylim(-1.2 * max_range, 1.2 * max_range)
    max_range = max(np.max(np.abs(x1s)), np.max(np.abs(x2s)))
    ax_orbit.set_xlabel("x position (m)")
    ax_orbit.set_ylabel("y position (m)")
    ax_orbit.set_title(f"Planetary Orbits E: {angleE * 180/np.pi :2f} M: {angleM * 180/np.pi :2f}")
    ax_orbit.grid(True)
    #ax_orbit.legend()

    # === Slider and TextBox ===
    slider_ax = plt.axes([0.2, 0.12, 0.6, 0.03])
    # time_slider = Slider(slider_ax, 'Time (years)', 0, t[-1], valinit=0, valstep=0.01)  # full time scale

    time_slider = Slider(slider_ax, 'Time (years)', 0, 0.1*t[-1], valinit=0, valstep=0.001)

    text_ax = plt.axes([0.83, 0.12, 0.1, 0.03])
    time_text = TextBox(text_ax, '', initial="0.00")

    fig.text(0.15, 0.06, 'Guess Angle (deg):', fontsize=10, ha='right', va='center')
    angle_input_ax = plt.axes([0.16, 0.05, 0.1, 0.04])
    angle_textbox = TextBox(angle_input_ax, '', initial="0.0")

    fig.text(0.15, 0.02, 'FOV (deg):', fontsize=10, ha='right', va='center')
    fov_input_ax = plt.axes([0.16, 0.01, 0.1, 0.04])
    fov_textbox = TextBox(fov_input_ax, '', initial="0.0")

    # === Update Function ===
    def update(val):
        idx = min(int(val / (t[1] - t[0])), len(x1s) - 1)

        # Update orbit markers
        earth_marker.set_data([x1s[idx]], [y1s[idx]])
        mars_marker.set_data([x2s[idx]], [y2s[idx]])

        # F_mars_vector.set_data([x1s[idx],x1s[idx]+Fx_mars_theoretical_norm[idx]*max_range],[y1s[idx], y1s[idx]+Fy_mars_theoretical_norm[idx]*max_range])
        F_sun_vector.set_data ([x1s[idx],x1s[idx]+Fx_sun_norm[idx]*max_range],[y1s[idx], y1s[idx]+Fy_sun_norm[idx]*max_range])
        F_net_vector.set_data ([x1s[idx],x1s[idx]+Fx_net_norm[idx]*max_range],[y1s[idx], y1s[idx]+Fy_net_norm[idx]*max_range])
        F_mars_theoretical_vector.set_data([x1s[idx],x1s[idx]+Fx_mars_theoretical_norm[idx]*max_range], [y1s[idx], y1s[idx]+Fy_mars_theoretical_norm[idx]*max_range])        
        # F_mars_theoretical_sim_vector.set_data([x1s[idx],x1s[idx]+Fx_mars_theoretical_sim_norm[idx]*max_range], [y1s[idx], y1s[idx]+Fy_mars_theoretical_sim_norm[idx]*max_range])        

        # Update text box
        time_text.set_val(f"{val:.2f}")

        fig.canvas.draw_idle()

    # === TextBox Submit ===
    def submit_text(text):
        try:
            val = float(text)
            val = max(0, min(val, t[-1]))
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
        new_val = current_val + 0.01  # years
        if new_val >= t[-1]:
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
        mars_orbit_line.set_visible(mars_visible[0])
        toggle_button.label.set_text('Show Mars' if not mars_visible[0] else 'Hide Mars')
        fig.canvas.draw_idle()

    fov = 5.0
    last_angle = 0.0

    toggle_button.on_clicked(toggle_mars)
    angle_toggle_button.on_clicked(toggle_angle)

    plt.show()

    # Initialize plot
    update(0)

plot_everything()
