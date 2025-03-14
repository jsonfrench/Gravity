import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.integrate import odeint
import samples

def one_body(IVP, t, G, M): 
    
    x, y, vx, vy = IVP  # unpack initial value problem 

    r = np.sqrt(x**2 + y**2)    # distance from sun to body

    ax = -G * M * x / r**3  # calculute force of sun on the body
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

def orbital_error(theoretical, actual, index_observation: int) -> tuple:
    theor_x = theoretical[:,0]
    theor_y = theoretical[:,1]
    actual_x = actual[:,0]
    actual_y = actual[:,1]

    x_error = np.abs(theor_x[index_observation] - actual_x[index_observation])
    y_error = np.abs(theor_y[index_observation] - actual_y[index_observation])

    L2_error = np.sqrt(x_error**2 + y_error**2)

    return (x_error,y_error,L2_error)

def orbital_distance(actual, index_observation: int) -> tuple:
    actual_x1 = actual[:,0]
    actual_y1 = actual[:,1]
    actual_x2 = actual[:,4]
    actual_y2 = actual[:,5]

    x_dist = np.abs(actual_x2[index_observation] - actual_x1[index_observation])
    y_dist = np.abs(actual_y2[index_observation] - actual_y1[index_observation])

    L2_dist = np.sqrt(x_dist**2 + y_dist**2)

    return (L2_dist)

def get_years(thetas) -> np.array:
    years = ([])  # list of time points when new year occurs
    for i in range ((len(thetas))-1):
        if thetas[i] > thetas[i+1]:
            years = np.append(years, (i+0.5) / samples_per_time)
    return years

def get_year_lengths(years) -> np.array:
    year_lengths = np.array([])
    for i in range(len(years)-1):
        year_lengths = np.append(year_lengths, years[i+1] - years[i]) # add year lengths to list
    return year_lengths

def get_yearly_distance(actual, years) -> tuple:
    
    new_year_dist = np.array([])
    for year in years:
        new_year_dist = np.append(new_year_dist, orbital_distance(actual, int(year)))

    avg_dist_per_year = np.array([])
    for i in range(len(years)-1):
        sum = 0 
        for j in range(int(years[i]), int(years[i+1])):
            sum += orbital_distance(actual, int(j))
        avg = sum / (years[i+1] - years[i])
        avg_dist_per_year = np.append(avg_dist_per_year, avg)

    return (new_year_dist, avg_dist_per_year)

def get_yearly_error(theoretical, actual, years) -> np.array:
    
    new_year_error = np.array([])
    for year in years:
        new_year_error = np.append(new_year_error, orbital_error(theoretical, actual, int(year)))

    return new_year_error


# Define Constants
G                   = 6.67 * 10 ** (-11)
samples_per_time    = 128
time_steps          = 500000
t                   = np.linspace(0, time_steps, time_steps*samples_per_time)
freq                = np.fft.rfftfreq(len(t), 1/samples_per_time)  # frequencies to be used by the discrete fourier transform 

IVP = samples.decreasing_dist[6]
# IVP = [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.99, 100000, 1, 1]
# IVP = samples.decreasing_dist[7]
# IVP = samples.decreasing_dist[8]

x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = IVP
actual_solution             = odeint(two_body, IVP[0:8], t, args=(G, IVP[8], IVP[9], IVP[10]))
actual_solution_thetas      = np.arctan2(actual_solution[:,1], actual_solution[:,0])
actual_solution_thetas_2    = np.arctan2(actual_solution[:,5], actual_solution[:,4])    # second body
theoretical_solution        = odeint(one_body, [x1, y1, vx1, vy1], t, args=(G, m0))
theoretical_solution_thetas = np.arctan2(theoretical_solution[:,1], theoretical_solution[:,0])
index_max                   = time_steps #* samples_per_time
min_frequency               = 0 #40
max_frequency               = 100 


# ======== year length + L2 distance

years = get_years(actual_solution_thetas)
year_lengths = get_year_lengths(years)
yearly_dist = get_yearly_distance(actual_solution, years)[0]
yearly_avg_dist = get_yearly_distance(actual_solution, years)[1]

bruh = plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.plot(np.linspace(2,len(years), len(years)-1), year_lengths)
plt.title("Body 1 Year length")
plt.ylabel("length of year")
plt.xlabel("year")

plt.subplot(2,2,3)
plt.plot(np.linspace(2,len(yearly_avg_dist), len(yearly_avg_dist)), yearly_avg_dist, color="lime", label="average distance")
plt.plot(np.linspace(2,len(yearly_dist), len(yearly_dist)), yearly_dist, color="green", label="new year distance")
plt.title("L2 Distance per year")
plt.ylabel("distance")
plt.xlabel("year")
plt.legend(loc="upper left")

plt.subplot(2,2,(2,4))
year_length_fft = np.abs(np.fft.rfft(year_lengths))
year_freq = np.fft.rfftfreq(year_lengths.size, 1)
min_freq = int(0*time_steps)
max_freq = int(10*time_steps)
plt.plot(year_freq[min_freq:max_freq], (year_length_fft/(year_length_fft.size/2))[min_freq:max_freq])
plt.title("Year length fft")
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
plt.show()

# =========== 3d plot stuff

# max_circular_dist = 2*IVP[4] - (IVP[4]-IVP[0])

# ax = plt.figure().add_subplot(projection='3d')

# z_resolution = 100
# increment = time_steps * samples_per_time // z_resolution
# for i in range(z_resolution):
#     # print(f"plotting {i+1}/{z_resolution}")
#     x = freq[min_frequency:max_frequency]
#     y =  np.abs(np.fft.rfft(actual_solution[:, 0][:increment * (i+1)]))[min_frequency:max_frequency]

#     slice_distance = orbital_distance(actual_solution, (increment * (i+1)-1)) # L2 Distance at time value corresponding to slice
#     normalized_color = int(slice_distance / max_circular_dist * 255)   # converts value of 0-L2_max to 0-255

#     # ax.plot(x, y, zs=i, zdir='x', color=mpl.colormaps['viridis'].colors[int((255/z_resolution) * (i+1))]) # gradient
#     ax.plot(x, y, zs=i, zdir='x', color=mpl.colormaps['viridis'].colors[normalized_color])

#     print(f"plotting {i+1}/{z_resolution} ... time {increment * (i+1)} slice_distance:{slice_distance} -> {normalized_color}")

# ax.set_xlabel('Time')
# ax.set_ylabel('Frequency')
# ax.set_zlabel('Amplitude')
# ax.view_init(elev=20., azim=-35, roll=0)
# plt.show()

