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
samples_per_time = 128
time_steps = 100000
t = np.linspace(0, time_steps, time_steps*samples_per_time) # array of time points to solve the system at
freq = np.fft.rfftfreq(len(t), 1/samples_per_time)  # frequencies to be used by the discrete fourier transform 

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

def plot_two_body_orbit(solution):

    two_body_solution = solution

    two_body_x1 = two_body_solution[:, 0]
    two_body_y1 = two_body_solution[:, 1]
    two_body_x2 = two_body_solution[:, 4]
    two_body_y2 = two_body_solution[:, 5]

    plt.plot(two_body_x1,two_body_y1, label="body 1")
    plt.plot(two_body_x2,two_body_y2, label = "body 2")

def plot_fourier_difference(solution_1, solution_2, title):

    # set bounds of plot
    min_frequency = 0 #40
    max_frequency = 100 #50
    truncated_range = freq[min_frequency:max_frequency]

    # get x part of the solution and fourier transform it
    solution_1_values = solution_1[:, 0]
    solution_2_values = solution_2[:, 0]
    solution_1_fourier = np.abs(np.fft.rfft(solution_1_values))[min_frequency:max_frequency]
    solution_2_fourier = np.abs(np.fft.rfft(solution_2_values))[min_frequency:max_frequency]

    # plot lines
    plt.plot(truncated_range,solution_1_fourier, color="blue", label="theoretical orbit")
    plt.plot(truncated_range,solution_2_fourier, color="orange", label="actual orbit")
    plt.plot(truncated_range,solution_1_fourier - solution_2_fourier, color="red", label="real difference")
    plt.plot(truncated_range,(solution_1_fourier - solution_2_fourier) * -1j, color="black", label="imaginary difference")

    plt.title(title)
    plt.legend(loc="upper right")
    # plt.show()

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


# examples = samples.eliptical
# examples = samples.increasing_vx
# examples = samples.decreasing_dist
# examples = samples.increasing_both_mass
# examples = samples.increasing_2nd_body_mass
# examples = samples.decreasing_dist_opposing_orbits
# examples = samples.increasing_both_masses_and_distances

# examples = []
# num=5
# for i in range(num):
#     examples.append(
#         [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * (1.0 + i*(0.4/(num+1))), 100000, 1, 1], 
#     )

# examples = [examples[2]]

# examples = [[1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 1.4, 100000, 1, 1]]
# examples = [[1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.99, 100000, 1, 1]] # messier coil orbit
examples = [samples.decreasing_dist[7]]

categories=[]
peak_heights=[] 
peak_widths=[]


masses=np.array([])
distances = np.array([])
eccentricities = np.array([])

labels = constants.labels

# fig = plt.figure(figsize=(8,6))

######################
# Main Examples Loop #
######################

for i in range(len(examples)): 
    
    print(f"simulating: {i+1}/{len(examples)}") # log progress on completing simulations 

    # set up and run  simulation
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = examples[i]
    two_body_solution = odeint(two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
    one_body_solution = odeint(one_body, [x1, y1, vx1, vy1], t, args=(G, m0))

    plt.subplot(len(examples), 1, i+1)  # Make a plot with 1 column and i rows

    # Add chart title
    if i == 0:
        title = "Base orbit"
    else:
        title = ""
        for j in range(len(examples[i])):
            title += "" if examples[i][j] == examples[0][j] else f"{labels[j]}={examples[i][j]} " # add parameter to title if it doesnt match base state

    # run functions
    plt.subplot(1,3,1)
    plot_two_body_orbit(two_body_solution)
    plt.title("Two Body Orbit")
    plt.ylabel("y_position")
    plt.xlabel("x_position")
    plt.legend()
    # plt.show()

    # plot_fourier_difference(one_body_solution, two_body_solution, title)
    # plt.plot(t, np.abs(np.fft.rfft(two_body_solution[:, 0])), color="orange")

    plt.subplot(1,3,2)
    plt.plot(t, (two_body_solution[:, 0]), color="blue")
    plt.title("X Position of First Body")
    plt.xlabel("Time")
    plt.ylabel("X Position")
    # plt.show()

    plt.subplot(1,3,3)
    plt.plot(np.linspace(0,100,100), np.abs(np.fft.rfft(two_body_solution[:, 0]))[0:100], color="orange")
    plt.title("Fourier Transform of X Position")

    plt.show()

    mark_peaks(one_body_solution, two_body_solution)
    # plot_peak_width(two_body_solution, title)

    # store data regarding variation in states
    eccentricities = np.append(eccentricities,[examples[i][7] / examples[0][7]])
    distances = np.append(distances,examples[i][4]-examples[i][0])
    masses = np.append(masses,examples[i][10])
    categories.append(title)

plt.tight_layout()  
plt.show()  # show stuff we plotted in the examples loop

#################
# Scatter plots #
#################

for i in range(len(peak_heights)):

    plt.scatter(masses[i],peak_heights[i])
    print(f"({masses[i]},{peak_heights[i]})")
    plt.xlabel = "mass"
    plt.ylabel = "peak height"
    # plt.scatter(masses[i],peak_widths[i])
    # print(f"({masses[i]},{peak_widths[i]})")
    # plt.xlabel = "mass"
    # plt.ylabel = "peak width"

    # plt.scatter(distances[i],peak_heights[i], c="orange")
    # print(f"({distances[i]},{peak_heights[i]})")
    # plt.scatter(distances[i],peak_widths[i])
    # print(f"({distances[i]},{peak_widths[i]})")

    # plt.scatter(eccentricities[i],peak_heights[i],c="orange")

# plt.plot(masses,1791.87136*masses + 951.85075)
# plt.plot(distances,2468.88*distances**-1.76386)
# print("peak heights:", peak_heights)
# print("peak widths:", peak_widths)

plt.show()  # show scatter plot of (mass/dist/vel) vs. (height/width)

# Create the bar chart
plt.bar(categories, peak_heights, color='orange')
print(peak_heights)

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('peak_heights')
plt.title('Height of Max Peak')

# Show the plot
plt.show()
