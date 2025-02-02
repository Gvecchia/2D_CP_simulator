# 2D SEARCHER STEM SIMULATOR MAIN
# Description: Integrate the equations for sensing activity and for the
#              mechanics. Then reconstruct the position of the stem in the plane
################################################################################
from __future__ import print_function
from fenics import *
from IPython import get_ipython
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import math
from classes import plant
from classes import plant_manager
import pandas as pd
################################################################################
# MODEL PARAMETERS
g   = 73231257600          # Gravity (m day^-2)
G0  = 0.12                 # Extension parameter (day^-1)
L0  = 0.17                 # Initial length (m)
Lg  = 0.17                 # Extension zone (m)
L_final = 0.844            # Final length (m)
theta0 = (np.pi)/2         # Initial angle
kappa0 = 0                 # Initial virtual curvature

alpha0  = 0.001            # Horizontal sensing
beta0   = 0.004            # Vertical sensing
gamma0  = 0.004            # Proprioception
E = 3244.25110*1E6*((36*24*100)**2) # Young modulus[Kg m day^-2 * m^-2]
n_leav = 3                 # number of leaves
l_int = 0.13               # avarege length of a mature internode (m)
                           # !!! actually, this value is not uesd. The
                           # expression raises an error. The value is implemented directily 
                           # inside the expression. 

# TARGET REACH AND ORIENTATION
orientation_final = 0.2*np.pi/2  # Final ori. of the shoot wrt the horizontal line (rad)
reach_final = 0.79               # Final Reach of the shoot (m)


# NUMERICAL PARAMETERS
nx  = 2**10                         # Space discretization
T   = (L_final- L0)/(G0 * Lg)       # Final Time
nt  = 2**8                          # Time discretization
dt  = T / nt                        # Time step size

# Optimization
# Wanna_Optimize:
# 0: Don't optimize;
# 1: Optimise by fixing gamma and minimising with respect alpha and beta.
#    Then gamma varies in an interval of range given by range_opt, using first
#    a step of size 1, then a step of size 0.1;
# 2: Optimise by varying all the parameters at the same time.
wanna_optimize = 0 # 1: optimize; else: don't
range_opt = 0.01 # The range for the optimization research in gamma
range_sens = 0.01
################################################################################
my_plant = plant(g,G0,L0,Lg,L_final,
orientation_final,reach_final,E,n_leav,l_int,theta0,kappa0,alpha0,beta0,gamma0)

manager = plant_manager(my_plant,nx,dt,T)

if wanna_optimize == 1:
    gmax = gamma0 + range_opt
    gmin = gamma0 - range_opt
    step = range_opt/10
    [param,err] = manager.OPT2(step,gmax,gmin,range_sens)
    min_err = min(err)
    index = err.index(min_err)
    param_opt = param[index]
    my_plant.alpha0 = param_opt[0]
    my_plant.beta0 = param_opt[1]
    my_plant.gamma0 = param_opt[2]
    # Export to csv
    header = np.array(['alpha','beta','gamma'])
    param_str = []
    for i in param_opt:
        num = str(i)
        num = num.replace('.',',')
        param_str.append(num)
    exp_tab = np.array([header,param_str])
    df = pd.DataFrame(exp_tab)
    df.to_csv('sensing_parameters.csv',sep=';',header=False, index=False)
elif wanna_optimize == 2:
    [param_opt] = manager.OPT3p2()
    my_plant.alpha0 = param_opt[0]
    my_plant.beta0 = param_opt[1]
    my_plant.gamma0 = param_opt[2]
    # Export to csv
    header = np.array(['alpha','beta','gamma'])
    exp_tab = np.array([header,param_opt])
    df = pd.DataFrame(exp_tab)
    df.to_csv('sensing_parameters.csv',sep=';',header=False, index=False)

[xall_co, yall_co, sall_co, Rall_co,
LD_leavall_co, VD_stemall_co,kaall_co] = manager.solution_computation()

reach = manager.compute_reach(xall_co,yall_co)
orientation = manager.compute_orientation(xall_co,yall_co)

# y-x Ratio
rat = yall_co[-1,:]/xall_co[-1,:]

# Export coordinates stem
df = pd.DataFrame(xall_co)
df.to_csv('xall.csv',sep=';',header=False, index=False)
df = pd.DataFrame(yall_co)
df.to_csv('yall.csv',sep=';',header=False, index=False)

# Plot
timeline = np.linspace(0,T,np.size(xall_co,1))
# Stem
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111)
ax.set_aspect('equal')
plt.plot(xall_co[:,::10],yall_co[:,::10])
plt.plot(xall_co[:,-1],yall_co[:,-1])
plt.xlabel('x axis [m]', fontsize=30)
plt.ylabel('y axis [m]', fontsize=30)
plt.grid()
ax.tick_params(axis='both', which='major', labelsize=20)
plt.show()

# Reach
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111)
plt.plot(timeline,reach,'k-', label = 'Simulated reach')
plt.plot(timeline[-1], reach_final, 'ro', label = 'Measured reach')
plt.xlabel('day', fontsize=30)
plt.ylabel('reach [m]', fontsize=30)
plt.grid()
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(prop = {'size': 20})
plt.show()

# Orientation
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111)
plt.plot(timeline,orientation,'k-', label = 'Simulated orientation')
plt.plot(timeline[-1],orientation_final,'ro', label = 'Measured orientation')
plt.xlabel('day', fontsize=30)
plt.ylabel('Orientation [rad]', fontsize=30)
plt.grid()
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(prop = {'size': 20})
plt.show()

# Mass
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111)
for i in range(0,np.size(xall_co,1),1):
    space = sall_co[:,i]
    mass = Rall_co[:,i]
    plt.plot(space,mass)
plt.xlabel('position (m)', fontsize=30)
plt.ylabel('mass (kg)', fontsize=30)
plt.grid()
plt.show()
print('Final Mass:',Rall_co[0,-1])

# Mass Leaves
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111)
space = sall_co[:,-1]
LD_leaves = LD_leavall_co[:,-1]
plt.plot(space,LD_leaves)
plt.xlabel('position (m)', fontsize=30)
plt.ylabel('Mass_leaves', fontsize=30)
plt.grid()
plt.show()

# Volume Density
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111)
space = sall_co[:,-1]
Volume_density = VD_stemall_co[:,-1]
plt.plot(space,Volume_density)
plt.xlabel('position (m)', fontsize=30)
plt.ylabel('Volume density', fontsize=30)
plt.grid()
plt.show()

# Intrinsic Curvature
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111)
space = sall_co[:,-1]
Icurvature = kaall_co[:,:]
plt.plot(space,Icurvature)
plt.xlabel('position (m)', fontsize=30)
plt.ylabel('Curvature', fontsize=30)
plt.grid()
plt.show()
