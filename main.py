# French 2.23.25

import numpy as np
from scipy.integrate import odeint
import scipy as sci
import matplotlib.pyplot as plt
import samples
import constants

# Constatnts
G = constants.G

# Simulation parameters
samples_per_time    = 128
time_steps          = 10000
t                   = np.linspace(0, time_steps, time_steps*samples_per_time) # array of time points to solve the system at
freq                = np.fft.rfftfreq(t.size+1, 1/samples_per_time)  # frequencies to be used by the discrete fourier transform 

#############
# Functions #
#############

def one_body(IVP, t, G, M): 
    
    x, y, vx, vy = IVP  # unpack initial value problem 

    r = np.sqrt(x**2 + y**2)    # distance from sun to body

    ax = -G * M * x / r**3  # calculute force of sun on the body
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]

def two_body(IVP, t, G, M, m1, m2): 
    
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP    # unpack initial value problem

    r1 = np.sqrt(x1**2 + y1**2) # distance of 1st body to sun
    r2 = np.sqrt(x2**2 + y2**2) # distance of 2nd body to sun
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2)    # distance between 1st and 2nd body

    ax1 = (-G * M * x1 / r1**3) + (-G * m2 * (x2-x1) / d**3)
    ay1 = (-G * M * y1 / r1**3) + (-G * m2 * (y2-y1) / d**3)
    ax2 = (-G * M * x2 / r2**3) + (-G * m1 * (x1-x2) / d**3)
    ay2 = (-G * M * y2 / r2**3) + (-G * m1 * (y1-y2) / d**3)

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

def three_body(IVP, t, G, M, m1, m2): 
    
    x0, y0, vx0, vy0, x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP    # unpack initial value problem

    r1 = np.sqrt((x1-x0)**2 + (y1-y0)**2) # distance of 1st body to sun
    r2 = np.sqrt((x2-x0)**2 + (y2-y0)**2) # distance of 2nd body to sun
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2)    # distance between 1st and 2nd body

    ax0 = (G * m1 * x1 / r1**3) + (G * m2 * x2 / r2**3)
    ay0 = (G * m1 * y1 / r1**3) + (G * m2 * y2 / r2**3)
    ax1 = (-G * M * x1 / r1**3) + (-G * m2 * (x2-x1) / d**3)
    ay1 = (-G * M * y1 / r1**3) + (-G * m2 * (y2-y1) / d**3)
    ax2 = (-G * M * x2 / r2**3) + (-G * m1 * (x1-x2) / d**3)
    ay2 = (-G * M * y2 / r2**3) + (-G * m1 * (y1-y2) / d**3)

    return [vx0, vy0, ax0, ay0, vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

def compute_fft(solution) -> np.array:
    return np.abs(np.fft.rfft(solution))

def compute_complex_fft(solution) -> np.array:
    return np.fft.rfft(solution)

def compute_fourier_difference(fft_1, fft_2) -> tuple:

    real_difference = fft_1 - fft_2
    imaginary_difference = fft_1 - fft_2 * -1j

    return (real_difference, imaginary_difference)

def plot_three_body_orbit(two_body_solution) -> None:

    two_body_x0 = two_body_solution[:, 0]
    two_body_y0 = two_body_solution[:, 1]
    two_body_x1 = two_body_solution[:, 4]
    two_body_y1 = two_body_solution[:, 5]
    two_body_x2 = two_body_solution[:, 8]
    two_body_y2 = two_body_solution[:, 9]

    # plt.plot(two_body_x0,two_body_y0, label="Body 0")
    # plt.plot(two_body_x1,two_body_y1, label="Body 1")
    # plt.plot(two_body_x2,two_body_y2, label = "Body 2")

    plt.plot(t,two_body_y0, label="Body 0")
    plt.plot(t,two_body_y1, label="Body 1")
    plt.plot(t,two_body_y2, label = "Body 2")

    plt.legend()

def plot_two_body_orbit(two_body_solution) -> None:

    two_body_x1 = two_body_solution[:, 0]
    two_body_y1 = two_body_solution[:, 1]
    two_body_x2 = two_body_solution[:, 4]
    two_body_y2 = two_body_solution[:, 5]

    plt.plot(two_body_x1,two_body_y1, label="Body 1")
    plt.plot(two_body_x2,two_body_y2, label = "Body 2")
    plt.legend()

def plot_one_body_orbit(one_body_solution) -> None:

    one_body_x1 = one_body_solution[:, 0]
    one_body_y1 = one_body_solution[:, 1]

    plt.title("Theoretical orbit")
    plt.plot(one_body_x1,one_body_y1, label="Body 1")
    plt.legend()

def plot_fourier_difference(fft_1, fft_2, min_freq=0, max_freq=freq.size) -> None:

    difference = compute_fourier_difference(fft_1, fft_2)
    real_difference = difference[0]
    imaginary_difference = difference[1]

    # plot lines
    plt.plot(freq[min_freq:max_freq], (fft_1/(t.size/2))[min_freq:max_freq],                 color="blue",   label="theoretical orbit")
    plt.plot(freq[min_freq:max_freq], (fft_2/(t.size/2))[min_freq:max_freq],                 color="orange", label="actual orbit")
    plt.plot(freq[min_freq:max_freq], (real_difference/(t.size/2))[min_freq:max_freq],       color="red",    label="real difference")
    plt.plot(freq[min_freq:max_freq], (imaginary_difference/(t.size/2))[min_freq:max_freq],  color="black",  label="imaginary difference")

    plt.legend(loc="upper right")

def mark_peaks(solution_1, solution_2):

    # peak detection stuff
    one_body_solution_x = solution_1[:, 0]  # get x values of one body solution
    two_body_solution_x = solution_2[:, 0]  # get x values of two body solution
    one_body_fourier = np.abs(np.fft.rfft(one_body_solution_x)) # fft the x values of one body solution
    two_body_fourier = np.abs(np.fft.rfft(two_body_solution_x)) # fft the x values of two body solution

    height_threshold = 20000 # only detect peaks that are at least this tall

    one_body_peaks, _ = sci.signal.find_peaks(one_body_fourier, height=height_threshold)   # Find peaks in one body fourier
    two_body_peaks, _ = sci.signal.find_peaks(two_body_fourier, height=height_threshold)   # Find peaks in two body fourier
    print("two body peaks",two_body_peaks)
    print(freq[two_body_peaks])
    print(two_body_fourier[two_body_peaks])
    plt.scatter(freq[one_body_peaks], one_body_fourier[one_body_peaks], c='blue')  # Mark peaks on one body fourier
    plt.scatter(freq[two_body_peaks], two_body_fourier[two_body_peaks], c='orange')  # Mark peaks on two body fourier

    one_body_tallest_peak_index = one_body_peaks[np.argmax(one_body_fourier[one_body_peaks])]
    two_body_tallest_peak_index = two_body_peaks[np.argmax(two_body_fourier[two_body_peaks])] 
    plt.scatter(freq[one_body_tallest_peak_index], one_body_fourier[one_body_tallest_peak_index], c='lime', marker="x", s=100)  # Mark tallest peak on one body fourier
    plt.scatter(freq[two_body_tallest_peak_index], two_body_fourier[two_body_tallest_peak_index], c='red', marker="x", s=100)  # Mark tallest peak on two body fourier

    tallest_peak = freq[two_body_tallest_peak_index]
    print("tallest peak frequency:", tallest_peak)
    print("tallest peak period:", 1/tallest_peak)
    print("tallest peak period adjusted:", 1/tallest_peak * samples_per_time)

    # set bounds of values
    min_frequency = 0 #40
    max_frequency = 100 #50

    # Gather weights and values for weighted average
    frequencies = freq[min_frequency:max_frequency]
    weights = two_body_fourier[min_frequency:max_frequency]
    peak_frequencies = freq[two_body_peaks]
    peak_weights = two_body_fourier[two_body_peaks]
    print("frequencies:", frequencies)
    print("peak frequencies:", peak_frequencies)

    # divide weights by sum of weights
    weights = weights / np.sum(weights)
    peak_weights = peak_weights / np.sum(peak_weights)
    print("weights:", weights)
    print("peak weights:", peak_weights)

    # Compute weighted average
    weighted_average_frequency = np.sum(frequencies * weights) 
    weighted_average_peak_frequency = np.sum(peak_frequencies * peak_weights)
    
    print("weighted average frequency:", weighted_average_frequency)
    print("predicted average period:", 1/weighted_average_frequency)
    print("predicted average period t value:", 1/weighted_average_frequency * samples_per_time)

    print("weighted average of frequency peaks:", weighted_average_peak_frequency)
    print("predicted average period using peaks:", 1/weighted_average_peak_frequency)
    print("predicted average period t value using peaks:", 1/weighted_average_peak_frequency * samples_per_time)

    tallest_peak_difference = two_body_fourier[two_body_tallest_peak_index]-one_body_fourier[one_body_tallest_peak_index] # compute difference in max peaks
    peak_heights.append(tallest_peak_difference)  
    # peak_widths.append(sci.signal.peak_widths(np.abs(x_hat[:F_upper]), [tallest_peak_index]))

def get_peaks(fft, min_peak_height=0) -> np.array:

    peaks, _ = sci.signal.find_peaks(fft, height=min_peak_height)   # Find peaks in a fourier spectrum

    return peaks

def compute_weighted_average(fft) -> tuple:

    freqs = freq
    weights = fft
    weights = weights / np.sum(weights)

    peak_freqs = get_peaks(fft)
    peak_weights = fft[peak_freqs]
    peak_weights = peak_weights / np.sum(peak_weights)

    weighted_average_frequency = np.sum(freqs * weights) 
    weighted_average_peak_frequency = np.sum(peak_freqs * peak_weights)

    return (weighted_average_frequency, weighted_average_peak_frequency)

def plot_peaks(fft, min_height=0, min_freq=0, max_freq=freq.size, color="blue") -> None:
    
    peaks = get_peaks(fft[min_freq:max_freq], min_height) + min_freq # returns list of freq indecies where peaks are

    tallest_peak = peaks[np.argmax(fft[peaks])]

    plt.scatter(freq[peaks], fft[peaks], color = color, marker="o") 
    plt.scatter(freq[tallest_peak], fft[tallest_peak], color = color, marker="x") 

def plot_peak_width(solution, title):

    solution_values = solution[:, 0]
    solution_fourier = np.abs(np.fft.rfft(solution_values))

    min_peak_height = 2000

    peaks, _ = sci.signal.find_peaks(solution_fourier, height=min_peak_height)  
    tallest_peak_index = peaks[np.argmax(np.abs(solution_fourier[peaks]))]  # Index of tallest peak

    spike_base = 0.9  # how far down to check the width of the spike 0.0 -> top of spike 1.0 -> bottom of spike

    width_info = sci.signal.peak_widths(solution_fourier, [tallest_peak_index], rel_height=spike_base)
    plt.hlines(y=width_info[1][0], xmin=freq[int(width_info[2][0])], xmax=freq[int(width_info[3][0])], color="lime",)

    width = width_info[3][0] - width_info[2][0] # width in frequency bins
    peak_widths.append(width)

def compare_orbits(examples, min_freq=0, max_freq=freq.size, graph="orbit") -> None: 

    for i in range(len(examples)):     

        print(f"simulating: {i+1}/{len(examples)}") # log progress on completing simulations 

        # set up and run  simulation
        x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = examples[i]
        two_body_solution = odeint(two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
        one_body_solution = odeint(one_body, [x1, y1, vx1, vy1], t, args=(G, m0))

        two_body_fourier = compute_complex_fft(two_body_solution[:, 0])   #fft of x values
        one_body_fourier = compute_complex_fft(one_body_solution[:, 0])   #fft of theoretical x values

        plt.subplot(len(examples), 1, i+1)  # Make a plot with 1 column and i rows

        # Add chart title
        labels = constants.labels
        if i == 0:
            title = "Base orbit"
        else:
            title = ""
            for j in range(len(examples[i])):
                title += "" if examples[i][j] == examples[0][j] else f"{labels[j]}={examples[i][j]} " # add parameter to title if it doesnt match base state
        plt.title(title)

        if graph == "fourier": 
            # compare fft
            plot_fourier_difference(two_body_fourier, one_body_fourier, min_freq, max_freq) 
        elif graph == "peaks": 
            # compare peaks
            plt.plot(freq[min_freq:max_freq], np.abs(two_body_fourier/(t.size/2))[min_freq:max_freq], color="orange")
            plot_peaks(np.abs(two_body_fourier/(t.size/2)), min_freq=min_freq, max_freq=max_freq, color="red")
        elif graph == "orbit": 
            # compare orbits
            plot_two_body_orbit(two_body_solution)

    plt.tight_layout()
    plt.show()

def compute_correlation(examples, independent_variable) -> None: # independent variable = (mass, distance, velocity)

    peak_heights=np.array([])
    independent_variable_values = np.array([])

    for i in range(len(examples)):     

        print(f"simulating: {i+1}/{len(examples)}") # log progress on completing simulations 

        # set up and run  simulation
        x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = examples[i]
        two_body_solution = odeint(two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
        two_body_fourier = compute_complex_fft(two_body_solution[:, 0])   #fft of x values

        peaks = get_peaks(two_body_fourier)
        tallest_peak_index = peaks[np.argmax(np.abs(two_body_fourier[peaks]))]  # Index of tallest peak

        # store data regarding variation in states
        if independent_variable == "mass":    
            independent_variable_values = np.append(independent_variable_values,examples[i][10])    # add mass data
        elif independent_variable == "distance":    
            independent_variable_values = np.append(independent_variable_values,examples[i][4]-examples[i][0])  # add distance data
        elif independent_variable == "eccentricity":    
            independent_variable_values = np.append(independent_variable_values,[examples[i][7] / examples[0][7]])  # add eccentricity data

        peak_heights = np.append(peak_heights, two_body_fourier[tallest_peak_index])

    for i in range(len(peak_heights)):
        # correlation with peak heights
        plt.scatter(independent_variable_values[i],peak_heights[i])
        print(f"({independent_variable_values[i]},{peak_heights[i]})")

    plt.xlabel(independent_variable.title())
    plt.ylabel("Peak Height")
    plt.title(f"Correlation between FFT Peak and {independent_variable.title()}")
    plt.show()  # show scatter plot of (mass/dist/vel) vs. (height/width)

def compute_one_body_energy(solution):


    v = np.sqrt(solution[:,2]**2 + solution[:,3]**2)
    m1 = examples[0][9]
    kinetic_energy = 0.5 * m1 * v**2

    M = examples[0][8]
    x = solution[:,0]
    y = solution[:,1]
    r = np.sqrt(x**2 + y**2)    # distance from sun to body
    potential_energy = -(G * M * m1) / r

    total_energy = kinetic_energy + potential_energy

    print(total_energy)

    # plt.plot(t,total_energy, c="lime")
    # plt.plot(t,kinetic_energy, c="orange")
    # plt.plot(t,potential_energy, c="blue")
    # plt.xlabel("time")
    # plt.ylabel("energy")
    # plt.title("Energy in the system over time")

    plt.suptitle("Energy in the system over time")
    plt.subplot(1,3,1)
    plt.plot(t,total_energy, c="lime")
    plt.title("total energy")
    plt.subplot(1,3,2)
    plt.plot(t,kinetic_energy, c="orange")
    plt.title("kinetic energy")
    plt.subplot(1,3,3)
    plt.plot(t,potential_energy, c="blue")
    plt.title("potential energy")

    plt.show()

def compute_three_body_energy(solution):

    v0 = np.sqrt(solution[:,2]**2 + solution[:,3]**2)
    m0 = examples[0][12]
    kinetic_energy_0 = 0.5 * m0 * v0**2

    v1 = np.sqrt(solution[:,6]**2 + solution[:,7]**2)
    m1 = examples[0][13]
    kinetic_energy_1 = 0.5 * m1 * v1**2

    v2 = np.sqrt(solution[:,10]**2 + solution[:,11]**2)
    m2 = examples[0][14]
    kinetic_energy_2 = 0.5 * m2 * v2**2

    kinetic_energy = kinetic_energy_0 + kinetic_energy_1 + kinetic_energy_2 

    M = examples[0][12]
    x0 = solution[:,0]
    y0 = solution[:,1]
    x1 = solution[:,4]
    y1 = solution[:,5]
    x2 = solution[:,8]
    y2 = solution[:,9]

    r1 = np.sqrt((x1-x0)**2 + (y1-y0)**2)    # distance from b1 to sun
    r2 = np.sqrt((x2-x0)**2 + (y2-y0)**2)    # distance from b2 to sun
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2)     # distance from b1 to b2

    potential_energy_0 = -(G * m0 * m1) / r1 + -(G * m0 * m2) / r2
    potential_energy_1 = -(G * m1 * m0) / r1 + -(G * m1 * m2) / d
    potential_energy_2 = -(G * m2 * m0) / r2 + -(G * m2 * m1) / d

    potential_energy = potential_energy_0 + potential_energy_1 + potential_energy_2

    total_energy = kinetic_energy + potential_energy

    print(total_energy)

    plt.suptitle("Energy in the system over time")
    plt.subplot(1,3,1)
    plt.plot(t,total_energy, c="lime")
    plt.title("total energy")
    plt.subplot(1,3,2)
    plt.plot(t,kinetic_energy, c="orange")
    plt.title("kinetic energy")
    plt.subplot(1,3,3)
    plt.plot(t,potential_energy, c="blue")
    plt.title("potential energy")

    plt.show()


# examples = []
# num=5
# for i in range(num):
#     examples.append(
#         [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * (1.0 + i*(0.4/(num+1))), 100000, 1, 1], 
#     )

# examples = [examples[2]]

# examples = [[1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 1.4, 100000, 1, 1]]
# examples = [[1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.99, 100000, 1, 1]] # messier coil orbit
# examples = samples.decreasing_dist[6:9]
# examples = [[1, 0, 0, np.sqrt(G*100000/1) * np.sqrt(2) * 0.99999, 1.5, 0, 0, np.sqrt(G*100000/1.5), 100000, 1, 1]]
# examples = [[1, 0, 0, np.sqrt(G*100000/1), 1.5, 0, 0, np.sqrt(G*100000/1.5), 100000, 1, 1]]
# examples = [[1, 0, 0, np.sqrt(G*100000/1), 1.1, 0, 0, np.sqrt(G*100000/1.1), 100000, 1, 1]]

# examples = [[0, 0, 0, 0, 1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01), 100000, 1, 1]]
M_sun = 3310000
examples = [[0, 0, 0, 0, 1, 0, 0, np.sqrt(G*M_sun/1), -2, 0, 0, -np.sqrt(G*M_sun/2), M_sun, 1, 1]]

# examples = samples.decreasing_dist
# examples = samples.increasing_2nd_body_mass

# compare_orbits(examples, min_freq=20, max_freq=60, graph="orbit")

x0, y0, vx0, vy0, x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = examples[0]
three_body_solution = odeint(three_body, [x0, y0, vx0, vy0, x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
plot_three_body_orbit(three_body_solution)
plt.show()

compute_three_body_energy(three_body_solution)
