import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import Any
import scipy as sci

# establish array as datatype
NDArray = np.ndarray[Any, np.dtype[np.float64]]

G = 6.67 * 10 ** (-11)

samples_per_time = 128
time_steps = 100000 * 1
t = np.linspace(0, time_steps, time_steps * samples_per_time)
freq = np.fft.rfftfreq(len(t), 1/samples_per_time)  # frequencies to be used by the discrete fourier transform 

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
def plot_two_body_orbit_interactive(solution: NDArray, tmax: int = 1000, theoretical: NDArray = None, plot_error = False) -> None:
    if plot_error == True and theoretical.all() == None:
        print('you forgot to input the theoretical orbit so error plotting is ignored')
        plot_error = False

    # moved figure creation to local rather than global
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    
    # initial index for the plot and slider
    initial_index = 320320  # weighted avg of peaks
    initial_index = 319729  # weighted avg over truncated spectrun
    
    # Compute the index corresponding to tmax
    index_max = tmax * samples_per_time

    # Establish x and y positions for the two bodies
    two_body_x1 = solution[:, 0]
    two_body_y1 = solution[:, 1]
    two_body_x2 = solution[:, 4]
    two_body_y2 = solution[:, 5]
    two_body_fourier = compute_fourier(solution[:initial_index])

    # plot the initial orbits
    line1, = ax1.plot(two_body_x1[:initial_index], two_body_y1[:initial_index])    
    line2, = ax1.plot(two_body_x2[:initial_index], two_body_y2[:initial_index])
    scatter1 = ax1.scatter(two_body_x1[initial_index], two_body_y1[initial_index], color='blue')
    scatter2 = ax1.scatter(two_body_x2[initial_index], two_body_y2[initial_index], color='orange')
    if plot_error == True:
        error = orbital_error(theoretical=one_body_solution, actual=two_body_solution, index_observation=initial_index)
        plot_current_error(error=error, current_index=initial_index, ax=ax1)
    fourier, = ax2.plot(two_body_fourier[0][:initial_index], two_body_fourier[1][:initial_index], color="orange")

    #ax.legend()

    # slider to control time
    axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
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
        line1.set_xdata(two_body_x1[:index])
        line1.set_ydata(two_body_y1[:index])
        line2.set_xdata(two_body_x2[:index])
        line2.set_ydata(two_body_y2[:index])

        scatter1.set_offsets([two_body_x1[index], two_body_y1[index]])
        scatter2.set_offsets([two_body_x2[index], two_body_y2[index]])

        two_body_fourier = compute_fourier(solution[:index])
        print(f"solution[:{index}]")
        # fourier.set_xdata(two_body_fourier[0][:index])
        # fourier.set_ydata(two_body_fourier[1][:index])
        fourier.set_xdata(two_body_fourier[0])
        fourier.set_ydata(two_body_fourier[1])

        height_threshold = 2000
        peaks, _ = sci.signal.find_peaks(two_body_fourier[1], height=height_threshold)   # Find peaks in two body fourier
        # print("peaks:",peaks)
        tallest_peak_index = peaks[np.argmax(two_body_fourier[1][peaks])]   
        # print("tallest peak index",tallest_peak_index)
        tallest_peak_height = two_body_fourier[1][tallest_peak_index]
        # print("tallest peak height", tallest_peak_height)

        ax2.set_ylim(-0.05* tallest_peak_height , tallest_peak_height + 0.05 * tallest_peak_height)  
        # print("tallst peak height:",tallest_peak_height)


        if plot_error == True:
            error = orbital_error(theoretical=one_body_solution, actual=two_body_solution, index_observation=index)
            plot_current_error(error=error, current_index = index, ax=ax1)


        fig.canvas.draw_idle()

    # update when slider changes
    time_slider.on_changed(update)

    # add reset button
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        time_slider.reset()

    button.on_clicked(reset)

    plt.show()

# method that will add the current orbital error to the interactive 
# plot of the two planets' orbits
def plot_current_error(error: tuple, current_index: int, ax: plt.Axes) -> None:
    for text in ax.texts:
        text.remove()

    # unpack error
    x_error, y_error, L2_error = error

    error_info = f'current \n x_err = {round(x_error,6)} \n y_err = {round(y_error,6)} \n L2_err = {round(L2_error,6)}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left
    ax.text(0.05, 0.95, error_info, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)


# method that computes the error in the x- and y-directions as well as the 
# L2 error between the theoretical and actual orbits of the known planet
# at a specific time.  Takes in the theoretical and actual orbital arrays 
# (in [[x],[y]] format), the *index* at which the observation is made, and 
# returns a tuple containing (x_error, y_error, L2_error)
def orbital_error(theoretical: NDArray, actual: NDArray, index_observation: int) -> tuple:
    theor_x = theoretical[:,0]
    theor_y = theoretical[:,1]
    actual_x = actual[:,0]
    actual_y = actual[:,1]

    x_error = np.abs(theor_x[index_observation] - actual_x[index_observation])
    y_error = np.abs(theor_y[index_observation] - actual_y[index_observation])

    L2_error = np.sqrt(x_error**2 + y_error**2)

    return (x_error,y_error,L2_error)

def compute_fourier(solution_1):

    # set bounds of plot
    min_frequency = 0 #40
    max_frequency = 100 #50
    truncated_range = freq[min_frequency:max_frequency]

    # get x part of the solution and fourier transform it
    solution_1_values = solution_1[:, 0]
    solution_1_fourier = np.abs(np.fft.rfft(solution_1_values))[min_frequency:max_frequency]

    # plot lines
    # plt.plot(truncated_range,solution_1_fourier, color="blue", label="observed orbit")

    return(truncated_range, solution_1_fourier)




###############################################################
# TESTING plot_two_body_orbit_interactive
###############################################################

IVP = [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.99, 100000, 1, 1]

x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = IVP
two_body_solution = odeint(two_body, IVP[0:8], t, args=(G, IVP[8], IVP[9], IVP[10]))
one_body_solution = odeint(one_body, IVP[0:4], t, args=(G, IVP[8]))

plot_two_body_orbit_interactive(two_body_solution, tmax=100000-1, theoretical=one_body_solution, plot_error=True)

