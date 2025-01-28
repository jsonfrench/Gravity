# Jason French 1.15.24

import numpy as np
from scipy.integrate import odeint
from scipy.fft import fft
from scipy.fft import rfft
import matplotlib.pyplot as plt


# Constatnts
G = 6.674e-11

# Parameters
m0 = 100000     # Large mass (sun)
m1 = 1          # small mass (1st body)
m2 = 1          # small mass (2nd body)

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
# vy1 = np.sqrt(G*m0) * 0.9         # eliptical
vy1 = np.sqrt(G*m0) * 1         # circular
# vy1 = np.sqrt(G*m0) * 1.1         # eliptical
# vy1 = np.sqrt(G*m0) * np.sqrt(2)    # parabolic 
# vy1 = np.sqrt(G*m0) * np.sqrt(2) * 1.1   # hyperbolic
vx2 = 0        
# vy2 = vy1 * 1.5 # <-- cool curve
vy2 = np.sqrt(G*m0) * 0.5 * 1.3

samples_per_time = 10
time_steps = 100000
t = np.linspace(0, time_steps, time_steps*samples_per_time)

def dSdt(IVP, t, G, M, m1, m2): 
    
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP

    r1 = np.sqrt(x1**2 + y1**2) # distance of 1st body to sun
    r2 = np.sqrt(x2**2 + y2**2) # distance of 2nd body to sun
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2)    # distance between 1st and 2nd body

    ax1 = (-G * M * x1 / r1**3) + (-G * m2 * (x2-x1) / d**3)
    ay1 = (-G * M * y1 / r1**3) + (-G * m2 * (y2-y1) / d**3)
    ax2 = (-G * M * x2 / r2**3) + (-G * m1 * (x1-x2) / d**3)
    ay2 = (-G * M * y2 / r2**3) + (-G * m1 * (y1-y2) / d**3)

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))

x1 = solution[:, 0]     # 1st body x values
y1 = solution[:, 1]     # 1st body y values
x2 = solution[:, 4]     # 2nd body x values
y2 = solution[:, 5]     # 2nd body y values

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()

freq = np.fft.rfftfreq(len(t), 1/samples_per_time)
fourier = np.fft.rfft(x1)
plt.plot(freq,np.abs(fourier))
plt.show()