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
# vy1 = np.sqrt(G*m0) * 0.9         # eliptical
vy1 = np.sqrt(G*m0) * 1         # circular
# vy1 = np.sqrt(G*m0) * 1.1         # eliptical
# vy1 = np.sqrt(G*m0) * np.sqrt(2)    # parabolic 
# vy1 = np.sqrt(G*m0) * np.sqrt(2) * 1.1   # hyperbolic
vx2 = 0        
vy2 = vy1


t = np.linspace(0, 100000, 1000000)

def dSdt(IVP, t, G, M, m1, m2): 
    
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP

    r1 = np.sqrt(x1**2 + y1**2)
    r2 = np.sqrt(x2**2 + y2**2)
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2)



    ax1 = (-G * M * x1 / r1**3) + (-G * M * (x2-x1) / d**3)
    ay1 = (-G * M * y1 / r1**3) + (-G * M * (y2-y1) / d**3)
    ax2 = (-G * M * x2 / r2**3) + (-G * M * (x2-x1) / d**3)
    ay2 = (-G * M * y2 / r2**3) + (-G * M * (y2-y1) / d**3)

    r1 = np.sqrt(x1**2 + y1**2)
    ax1 = -G * M * x1 / r1**3
    ay1 = -G * M * y1 / r1**3

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))

x1 = solution[:, 0]     # select all x values
y1 = solution[:, 1]     # select all y values
x2 = solution[:, 4]     # select all x values
y2 = solution[:, 5]     # select all y values

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()

fourier = fft(x1)
plt.plot(t,fourier)
plt.show()