import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import samples
from typing import Any

# establish array as datatype
NDArray = np.ndarray[Any, np.dtype[np.float64]]

G = 6.67 * 10 ** (-11)

samples_per_time = 128
time_steps = 100000 * 1
t = np.linspace(0, time_steps, time_steps * samples_per_time)

def one_body(IVP, t, G, M): 
    
    x, y, vx, vy = IVP

    r = np.sqrt(x**2 + y**2)

    ax = -G * M * x / r**3
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]

def two_body(IVP, t, G, M, m1, m2):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP

    r1 = np.sqrt(x1**2 + y1**2)  # distance of 1st body to sun
    r2 = np.sqrt(x2**2 + y2**2)  # distance of 2nd body to sun
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)  # distance between 1st and 2nd body

    ax1 = (-G * M * x1 / r1**3) + (-G * m2 * (x2 - x1) / d**3)
    ay1 = (-G * M * y1 / r1**3) + (-G * m2 * (y2 - y1) / d**3)
    ax2 = (-G * M * x2 / r2**3) + (-G * m1 * (x1 - x2) / d**3)
    ay2 = (-G * M * y2 / r2**3) + (-G * m1 * (y1 - y2) / d**3)

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

# method that will plot the two body orbits from t=0 to t=tmax;
# will be used to create an interactive plot with tmax as a slider.
# Must be called using the ipywidgets interact method.
# Takes ODE solution and tmax (converted to indices within) as arguments.
def plot_two_body_orbit_interactive(solution: NDArray, tmax: int = 1000, theoretical: NDArray = None, plot_error = False, plot_theoretical = False, plot_fourier = True, home_planet_mass: float = None) -> None:
    if plot_error == True and theoretical.all() == None:
        print('you forgot to input the theoretical orbit so error plotting is ignored')
        plot_error = False
    
    if plot_error == True and home_planet_mass == None:
        print('you forgot to input the home planet mass so error plotting is ignored')
        plot_error = False

    if plot_theoretical == True and theoretical.all() == None:
        print('you forgot to input the theoretical orbit so theoretical orbit plotting is ignored')
        plot_theoretical = False

    # moved figure creation to local rather than global
    # create two rows of plots if plot_fourier is active
    fig, ax = plt.subplots(plot_fourier + 1, 1, figsize=(8, 6))
    ax[0].set_xlim(-2, 2)
    ax[0].set_ylim(-2, 2)

    #ax[1].set_xlim(0,3)
    ax[1].set_ylim(0,1000)
    
    # initial index for the plot and slider
    initial_index = 1000
    
    # Compute the index corresponding to tmax
    index_max = tmax * samples_per_time

    # Establish x and y positions for the two bodies
    two_body_x1 = solution[:, 0]
    two_body_y1 = solution[:, 1]
    two_body_vx1 = solution[:, 2]
    two_body_vy1 = solution[:, 3]
    two_body_x2 = solution[:, 4]
    two_body_y2 = solution[:, 5]

    two_body_solution = np.array([two_body_x1,two_body_y1,two_body_vx1,two_body_vy1])

    if plot_theoretical == True:
        theor_x = theoretical[:,0]
        theor_y = theoretical[:,1]

    # plot the initial orbits
    home_orbit_actual, = ax[0].plot(two_body_x1[:initial_index], two_body_y1[:initial_index])
    mystery_orbit, = ax[0].plot(two_body_x2[:initial_index], two_body_y2[:initial_index])
    home_planet_actual = ax[0].scatter(two_body_x1[initial_index], two_body_y1[initial_index], color='blue')
    mystery_planet_actual = ax[0].scatter(two_body_x2[initial_index], two_body_y2[initial_index], color='orange')
    if plot_error == True:
        error = orbital_error(theoretical=one_body_solution, actual=two_body_solution, home_planet_mass=home_planet_mass, index_observation=initial_index)
        plot_current_error(error=error, current_index=initial_index, ax=ax[0])
    if plot_theoretical == True:
        t_segment = np.linspace(0,1,100)
        x_component_segment = t_segment * theor_x[initial_index] + (1-t_segment) * two_body_x1[initial_index]
        y_component_segment = t_segment * theor_y[initial_index] + (1-t_segment) * two_body_y1[initial_index]
        error_segment, = ax[0].plot(x_component_segment, y_component_segment, c='black')
        home_planet_theoretical = ax[0].scatter(theor_x[initial_index], theor_y[initial_index], c='black')
    if plot_fourier == True:
        x_hat = np.fft.rfft(two_body_x1[:initial_index])
        freq = np.fft.rfftfreq(len(two_body_x1[:initial_index]), 1./samples_per_time)
        x_hat_plot, = ax[1].plot(freq, np.abs(x_hat))

    #ax.legend()

    # slider to control time
    axtime = fig.add_axes([0.25, 0.5, 0.65, 0.03])
    time_slider = Slider(
        ax=axtime,
        label='time',
        valmin=0,
        valmax=index_max,
        valinit=initial_index
    )

    # adjust the plot based on the slider
    def update(val):
        index = int(time_slider.val)
        home_orbit_actual.set_xdata(two_body_x1[:index])
        home_orbit_actual.set_ydata(two_body_y1[:index])
        mystery_orbit.set_xdata(two_body_x2[:index])
        mystery_orbit.set_ydata(two_body_y2[:index])

        home_planet_actual.set_offsets([two_body_x1[index], two_body_y1[index]])
        mystery_planet_actual.set_offsets([two_body_x2[index], two_body_y2[index]])

        if plot_error == True:
            error = orbital_error(theoretical=one_body_solution, actual=two_body_solution, home_planet_mass=home_planet_mass, index_observation=index )
            plot_current_error(error=error, current_index = index, ax=ax[0])

        if plot_theoretical == True:
            x_component_segment = t_segment * theor_x[index] + (1-t_segment) * two_body_x1[index]
            y_component_segment = t_segment * theor_y[index] + (1-t_segment) * two_body_y1[index]
            error_segment.set_xdata(x_component_segment)
            error_segment.set_ydata(y_component_segment)
            home_planet_theoretical.set_offsets([theor_x[index], theor_y[index]])

        if plot_fourier == True:
            x_hat = np.fft.rfft(two_body_x1[:index])
            freq = np.fft.rfftfreq(len(two_body_x1[:index]), 1./samples_per_time)
            x_hat_plot.set_xdata(freq)
            x_hat_plot.set_ydata(np.abs(x_hat))

        fig.canvas.draw_idle()

    

    # update when slider changes
    time_slider.on_changed(update)

    # add reset button
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        time_slider.reset()

    button.on_clicked(reset)

    plt.tight_layout()
    plt.show()

# method that will add the current orbital error to the interactive 
# plot of the two planets' orbits
def plot_current_error(error: tuple, current_index: int, ax: plt.Axes) -> None:
    for text in ax.texts:
        text.remove()

    # unpack error
    x_error, y_error, L2_error, delta_p = error

    error_info = f'current \n x_err = {round(x_error,6)} \n y_err = {round(y_error,6)} \n L2_err = {round(L2_error,6)} \n delta_p = {round(delta_p,6)}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left
    ax.text(0.05, 0.95, error_info, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)


# method that computes the error in the x- and y-directions as well as the 
# L2 error between the theoretical and actual orbits of the known planet
# at a specific time.  Takes in the theoretical and actual orbital arrays 
# (in [[x],[y]] format), the *index* at which the observation is made, and 
# returns a tuple containing (x_error, y_error, L2_error)
def orbital_error(theoretical: NDArray, actual: NDArray, home_planet_mass: float, index_observation: int) -> tuple:
    theor_x = theoretical[:,0]
    theor_y = theoretical[:,1]
    actual_x = actual[0,:]
    actual_y = actual[1,:]
    actual_vx = actual[2,:]
    actual_vy = actual[3,:]

    x_error = np.abs(theor_x[index_observation] - actual_x[index_observation])
    y_error = np.abs(theor_y[index_observation] - actual_y[index_observation])

    L2_error = np.sqrt(x_error**2 + y_error**2)

    magnitude_delta_p = home_planet_mass * np.sqrt((actual_vx[index_observation]-actual_vx[0])**2 + (actual_vy[index_observation]-actual_vy[0])**2)

    return (x_error,y_error,L2_error, magnitude_delta_p)



###############################################################
# TESTING plot_two_body_orbit_interactive
###############################################################

examples1 = samples.decreasing_dist

x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = examples1[7]
two_body_solution = odeint(two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
one_body_solution = odeint(one_body, [x1, y1, vx1, vy1], t, args=(G, m0))

x_hat_full = np.fft.rfft(two_body_solution[:,0])
freq_full = np.fft.rfftfreq(len(two_body_solution[:,0]), 1./samples_per_time)
plt.plot(freq_full,np.abs(x_hat_full))

plot_two_body_orbit_interactive(two_body_solution, tmax=time_steps, theoretical=one_body_solution, plot_error=True, plot_theoretical=True, home_planet_mass = m1)
