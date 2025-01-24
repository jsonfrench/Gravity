# Jason French 1.15.24

import numpy as np
from scipy.integrate import odeint
import scipy as sci
import matplotlib.pyplot as plt
import statistics as stat



# Constants
G = 1

# Parameters
m0 = 1000000     # Large mass (sun)
m1 = 1          # small mass (planet)
m2 = 1

# Initial Values
x0 = 0         
y0 = 0         
x1 = 1        
y1 = 0      
x2 = 2
y2 = 0 
vx0 = 0        
vy0 = 0        
vx1 = 0        
#vy1 = np.sqrt(G*m1) * 0.9         # eliptical
#vy1 = np.sqrt(G*m1) * 1         # circular
vy1 = np.sqrt(G*m1) * 1        # eliptical
#vy1 = np.sqrt(G*m1) * np.sqrt(2)    # parabolic 
# vy1 = np.sqrt(G*m1) * np.sqrt(2) * 1.1   # hyperbolic
vx2 = 0
vy2 = np.sqrt(G*m1) * 1.1         # eliptical


def dSdt(IVP, t, G, M): 
    
    x, y, vx, vy = IVP

    r = np.sqrt(x**2 + y**2)

    ax = -G * M * x / r**3
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]

t_max = 10
samples_per_time = 50000

t = np.linspace(0, t_max, samples_per_time*t_max)

solution = odeint(dSdt, [x2, y2, vx2, vy2], t, args=(G, m1))

x = solution[:, 0]     # select all x values
y = solution[:, 1]     # select all y values

c = np.cos(200*2*np.pi*t)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('orbital position')

array = np.zeros(len(x))
for i in range(len(x)-1):
    array[i] = (t[i+1]-t[i])*(x[i+1]+x[i])

x_avg = (1/2*t_max)*np.sum(array)

x = x - x_avg
x_hat = np.fft.rfft(x)
freq = np.fft.rfftfreq(len(t), 1/samples_per_time)

F_upper = 10000
height = 200

peaks, properties = sci.signal.find_peaks(np.abs(x_hat[:F_upper]), height = height)

plt.subplot(1,2,2)
plt.plot(freq[:F_upper],np.abs(x_hat[:F_upper]))
plt.scatter(freq[peaks], np.abs(x_hat[peaks]), c='orange')
plt.text(x=250, y=100000, s=f'peak freq = {freq[peaks]}')
plt.xlabel('frequency')
plt.ylabel('magnitude')
plt.title('orbital x coordinate in frequency domain')
plt.tight_layout()
plt.show()
