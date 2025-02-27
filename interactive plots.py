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
def plot_two_body_orbit_interactive(solution: NDArray, tmax: int = 1000) -> None:
    
    # moved figure creation to local rather than global
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # initial index for the plot and slider
    initial_index = 500
    
    # Compute the index corresponding to tmax
    index_max = tmax * samples_per_time

    # Establish x and y positions for the two bodies
    two_body_x1 = solution[:, 0]
    two_body_y1 = solution[:, 1]
    two_body_x2 = solution[:, 4]
    two_body_y2 = solution[:, 5]

    # plot the initial orbits
    line1, = ax.plot(two_body_x1[:initial_index], two_body_y1[:initial_index])
    line2, = ax.plot(two_body_x2[:initial_index], two_body_y2[:initial_index])
    scatter1 = ax.scatter(two_body_x1[initial_index], two_body_y1[initial_index], color='blue')
    scatter2 = ax.scatter(two_body_x2[initial_index], two_body_y2[initial_index], color='orange')

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


###############################################################
# TESTING plot_two_body_orbit_interactive
###############################################################

examples1 = samples.decreasing_dist

x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = examples1[6]
two_body_solution = odeint(two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))


plot_two_body_orbit_interactive(two_body_solution, tmax=7500)
