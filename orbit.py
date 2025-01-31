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
# //// body 0
x0 = 0         
y0 = 0         
vx0 = 0        
vy0 = 0        
# //// body 1
x1 = 1
y1 = 0         
vx1 = 0        
vy1 = np.sqrt(G*m0/x1)         # circular
# //// body 2
x2 = 1.01    
y2 = 0         
vx2 = 0        
vy2 = np.sqrt(G*m0/x2) * 0.99

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

# /////////////////////////
def dSdt_one(IVP, t, G, M): 
    
    x, y, vx, vy = IVP

    r = np.sqrt(x**2 + y**2)

    ax = -G * M * x / r**3
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]

# solution1 = odeint(dSdt_one, [x1, y1, vx1, vy1], t, args=(G, m0))
# /////////////////////////

# solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))

# x1 = solution[:, 0]     # 1st body x values
# y1 = solution[:, 1]     # 1st body y values
# x2 = solution[:, 4]     # 2nd body x values
# y2 = solution[:, 5]     # 2nd body y values

# plt.plot(x1,y1)
# plt.plot(x2,y2)
# plt.show()

# freq = np.fft.rfftfreq(len(t), 1/samples_per_time) * 10
# fourier = np.fft.rfft(x1)
# plt.plot(freq,np.abs(fourier))
# fourier = np.fft.rfft(x2)
# plt.plot(freq,np.abs(fourier),"-", color="black")
# plt.show()


freq = np.fft.rfftfreq(len(t), 1/samples_per_time) * 10

# one body
x1, y1, vx1, vy1 = [1, 0, 0, np.sqrt(G*m0/1)]    
solution = odeint(dSdt_one, [x1, y1, vx1, vy1], t, args=(G, m0))
x1 = solution[:, 0]
y1 = solution[:, 1]
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 1")
plt.show()
plt.plot(x1,y1)
plt.title("Graph 1 - 1 body orbit")
plt.show()

# two bodies, coil orbits
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = [1, 0, 0, np.sqrt(G*m0/1), 1.01, 0, 0, np.sqrt(G*m0/1.01)]
solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
x1 = solution[:, 0]
y1 = solution[:, 1]
x2 = solution[:, 4]
y2 = solution[:, 5]
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 2")
plt.show()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.title("Graph 2 - coil orbits")
plt.show()

# two bodies, slightly messier coil orbits
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = [1, 0, 0, np.sqrt(G*m0/1), 1.01, 0, 0, np.sqrt(G*m0/1.01) * 0.99]
solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
x1 = solution[:, 0]
y1 = solution[:, 1]
x2 = solution[:, 4]
y2 = solution[:, 5]
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 3")
plt.show()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.title("Graph 3 - slightly messier coil")
plt.show()

# one body close to sun 

x1, y1, vx1, vy1 = [0.07, 0, 0, np.sqrt(G*m0/0.07) * 0.7]  
solution = odeint(dSdt_one, [x1, y1, vx1, vy1], t, args=(G, m0))
x1 = solution[:, 0]  
y1 = solution[:, 1]     
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 4")
plt.show()
plt.plot(x1,y1)
plt.title("Graph 4 - unstable 1 body")
plt.show()

# two bodies, large second mass
m2=100
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = [1, 0, 0, np.sqrt(G*m0/1), 1.01, 0, 0, np.sqrt(G*m0/1.01)]
solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
m2=1
x1 = solution[:, 0]
y1 = solution[:, 1]
x2 = solution[:, 4]
y2 = solution[:, 5]
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 5")
plt.show()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.title("Graph 5 - MASSIVE 2nd body")
plt.show()

# two bodies, minor x velocity 
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = [1, 0, 0.001, np.sqrt(G*m0/1), 2, 0, 0, np.sqrt(G*m0/2)]
solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
x1 = solution[:, 0]
y1 = solution[:, 1]
x2 = solution[:, 4]
y2 = solution[:, 5]
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 6")
plt.show()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.title("Graph 6 - threw some vx in there")
plt.show()

# two bodies, massive 1st body 
m1=1000
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = [1, 0, 0, np.sqrt(G*m0/1), 1.5, 0, 0, np.sqrt(G*m0/2)]
solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
x1 = solution[:, 0]
y1 = solution[:, 1]
x2 = solution[:, 4]
y2 = solution[:, 5]
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 7")
plt.show()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.title("Graph 7 - loooooow taaaaper faaaade ahh 1st body")
plt.show()

# two bodies, (less) massive 1st body 
m1=100
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = [1, 0, 0, np.sqrt(G*m0/1), 1.1, 0, 0, np.sqrt(G*m0/1.1)]
solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
x1 = solution[:, 0]
y1 = solution[:, 1]
x2 = solution[:, 4]
y2 = solution[:, 5]
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 8")
plt.show()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.title("Graph 8 - (less) massive 1st body")
plt.show()

# two bodies travelling opposite directions
m1=1
x1, y1, vx1, vy1, x2, y2, vx2, vy2 = [1, 0, 0, np.sqrt(G*m0/1), -1.01, 0, 0, np.sqrt(G*m0/1.01)]
solution = odeint(dSdt, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
x1 = solution[:, 0]
y1 = solution[:, 1]
x2 = solution[:, 4]
y2 = solution[:, 5]
s = solution[:, 0]     # 1st body x values
fourier = np.fft.rfft(s)
plt.plot(freq,np.abs(fourier))
plt.title("Graph 9")
plt.show()
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.title("Graph 9 - opposite direction orbits")
plt.show()
