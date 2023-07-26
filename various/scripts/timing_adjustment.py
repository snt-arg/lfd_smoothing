# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

T = 1.0
a = 0

def plot_function(func, start, end, num_points=100):
    x = np.linspace(start, end, num_points)
    y = func(x)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Function')
    plt.grid(True)
    plt.show()

def f(x):
    return x**3

def acc(t):
    return 0.1 * np.maximum(0, np.sin(t) + 1)

def g(t):
    a = acc(t)
    return 0.5*a*(t**2) + t/T

def fog(t):
    return f(g(t))

#%%

class Particle:
    
    def __init__(self, v0):
        self.v0 = v0
        self.reset()
        
    def reset(self):
        self.t = 0
        self.x = 0
         
    def acc(self, t):
        # return 0
        return 3 * np.maximum(0, np.sin(3*t) + 1)
    
    def positions(self,ts):
        self.reset()
        xs = []
        for t in ts:
            xs.append(self.position(t))
            
        return np.array(xs)

    def position(self,t):
            dt = t - self.t
            dx = 0.5*self.acc(t)*(dt**2) + self.v0*dt
            self.t = t
            self.x +=dx
            if self.x > 1: self.x = 1
            return self.x
        

#%%

particle = Particle(0.01)

plot_function(particle.positions, 0,10)

#%%

def f(x):
    return x**3

def fog(ts):
    xs = particle.positions(ts)
    return f(xs)

plot_function(fog, 0, 10)

#%%

a=0
plot_function(fog, 0, T)
a=0.1
plot_function(fog, 0, T)

#%%
a=100
plot_function(g, 0, 10)

#%%
plot_function(acc, 0, 10)

   