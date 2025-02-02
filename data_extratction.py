# DATA EXTRACTOR
# Description: Extratc data from .csv file for the main stem and for the leaves
#              mass distribution. The data are saved in a .csv file which is
#              used by the main script.
################################################################################
from __future__ import print_function
from IPython import get_ipython
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd
################################################################################
# MAIN STEM
################################################################################
# FILE OPENING
################################################################################
# Main stem
# The file "congui_main_stem.csv" has this structure
# header -> |Dis Apex (m)|Vol Den (Kg m^-3)|Leaf mass (Kg)|Flex Rig (N m^2)|...
# data ->   |     ...    |       ...       |     ...      |       ...      |

tab = []
with open("congui_p1s2_radius.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=';',quotechar="'" )
    for row in reader: # each row is a list
        tab.append(row)

tab = np.array(tab)
tab = tab[1:,:] # Take away first header row
row_size = np.size(tab,0)
col_size = np.size(tab,1)
tab2 = np.zeros((row_size,col_size)) # If I leave just tab, it doesn't work

# Convert "," in ".", then in float
for i in range(row_size):
    for j in range(col_size):
        num = tab[i,j]
        num = num.replace(',','.')
        tab2[i,j] = float(num)

d_ap = tab2[:,0]
vd_exp = tab2[:,1]
lld_exp = tab2[:,2]
fr_exp = tab2[:,4]
rad_exp = tab2[:,5]
rsp_exp = tab2[:,6]
################################################################################
# DATA ELABORATION
################################################################################
# Flexural Rigidity
def FR(d_ap,a,b,c):
    return a/(1 + np.exp(b*(c - d_ap)))

def eq_fr(x0):
    a = x0[0]
    b = x0[1]
    c = x0[2]
    val = 0
    for i in range(np.size(fr_exp)):
        val = val + (fr_exp[i] - FR(d_ap[i],a,b,c))**2
    return val

# First guess
x0 = [0.001, 1, 0.001]
err = 0
try:
    x_sol = minimize(eq_fr,x0)
    x0 = x_sol.x
except:
    err += 1
    print('"minimize" not working (Flexural Rigidity). Trying Curvefit.')
# Curve Fit
try:
    params, cv = curve_fit(FR, d_ap, fr_exp, x0)
    x0 = params
except:
    err += 1
    if err == 2:
        print('"Fitting failed (Flexural Rigidity).')
        print('Consider to change the fitting curves."')
        quit()
    print('"curve_fit" failed (Flexural Rigidity).')
    print('Considering the parameters of "minimize".')

a_fr = x0[0]
b_fr = x0[1]
c_fr = x0[2]

################################################################################
# Volume Density
# Consider the inverse to make the method work
VD = np.polyfit(d_ap, vd_exp,2)

print(VD)

a_vd = VD[0]
b_vd = VD[1]
c_vd = VD[2]

################################################################################
# Leaves Density
def LLD(d_ap,a,b,c):
    return a/(1 + np.exp(b*(c - d_ap)))

def eq_leav(x0):
    a = x0[0]
    b = x0[1]
    c = x0[2]
    val = 0
    for i in range(np.size(lld_exp)):
        val = val + (lld_exp[i] - LLD(d_ap[i],a,b,c))**2
    return val

# First guess
x0 = [0.01, 10, 0.1]
err = 0
try:
    x_sol = minimize(eq_leav,x0)
    x0 = x_sol.x
except:
    err += 1
    print('"minimize" not working (Leaves). Trying Curvefit.')
# Curve Fit
try:
    params, cv = curve_fit(LLD, d_ap, lld_exp, x0)
    x0 = params
except:
    err += 1
    if err == 2:
        print('"Fitting failed (Leaves).')
        print('Consider to change the fitting curves."')
        quit()
    print('"curve_fit" failed (Leaves).')
    print('Considering the parameters of "minimize".')

a_leav = x0[0]
b_leav = x0[1]
c_leav = x0[2]
################################################################################
# Radius
RAD = np.polyfit(d_ap, rad_exp,2)

print(RAD)

a_rad = RAD[0]
b_rad = RAD[1]
c_rad = RAD[2]
################################################################################
# Radius Speed
RSP = np.polyfit(d_ap, rsp_exp,2)

print(RSP)

a_rsp = RSP[0]
b_rsp = RSP[1]
c_rsp = RSP[2]

################################################################################
# Saving Parameters
################################################################################
header = np.array(['a_fr', 'b_fr', 'c_fr',
                    'a_vd', 'b_vd', 'c_vd',
                    'a_leav', 'b_leav', 'c_leav',
                    'a_rad','b_rad','c_rad',
                    'a_rsp','b_rsp','c_rsp'])

parameters = np.array([a_fr, b_fr, c_fr,
                       a_vd, b_vd, c_vd,
                      a_leav, b_leav, c_leav,
                      a_rad, b_rad, c_rad,
                      a_rsp, b_rsp, c_rsp])

exp_tab = np.array([header,parameters])
df = pd.DataFrame(exp_tab)
df.to_csv('parameters.csv',sep=';',header=False, index=False)
################################################################################
# PLOT
################################################################################
def FR_fun(d):
    return FR(d,a_fr,b_fr,c_fr)

VD_fun = np.poly1d(VD)

def LLD_fun(d):
    return LLD(d,a_leav,b_leav,c_leav)

RAD_fun = np.poly1d(RAD)
RSP_fun = np.poly1d(RSP)

# Plot
d_line = np.arange(min(d_ap), max(d_ap), 0.01)
fr_fit = []
vd_fit = []
lld_fit = []
rad_fit = []
rsp_fit = []

for i in d_line:
    fr_fit.append(FR_fun(i))
    vd_fit.append(VD_fun(i))
    lld_fit.append(LLD_fun(i))
    rad_fit.append(RAD_fun(i))
    rsp_fit.append(RSP_fun(i))

# Flexural Rigidity
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
plt.subplot(111)
plt.plot(d_line, fr_fit, 'k-')
plt.plot(d_ap, fr_exp,'ko')
plt.xlabel('Distance from the Apex [m]', fontsize=30)
plt.ylabel('Flexural Rigidity [$N \, m^2$]', fontsize=30)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()

# Volume Density
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
plt.subplot(111)
plt.plot(d_line, vd_fit, 'k-')
plt.plot(d_ap, vd_exp,'ko')
plt.xlabel('Distance from the Apex [m]', fontsize=30)
plt.ylabel('Volume Density [$Kg \, m^{-3}$]', fontsize=30)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()

# Linear Density Leaves
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
plt.subplot(111)
plt.plot(d_line, lld_fit, 'k-')
plt.plot(d_ap, lld_exp,'ko')
plt.xlabel('Distance from the Apex [m]', fontsize=30)
plt.ylabel('Leaves Mass [$Kg$]', fontsize=30)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()

# Radius
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
plt.subplot(111)
plt.plot(d_line, rad_fit, 'k-')
plt.plot(d_ap, rad_exp,'ko')
plt.xlabel('Distance from the Apex [m]', fontsize=30)
plt.ylabel('Radius [m]', fontsize=30)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()

# Radius speed
fig = plt.figure(figsize = (5,5))
fig.patch.set_alpha(0.0)
plt.subplot(111)
plt.plot(d_line, rsp_fit, 'k-')
plt.plot(d_ap, rsp_exp,'ko')
plt.xlabel('Distance from the Apex [m]', fontsize=30)
plt.ylabel('Radial expanzion speed [$m \, day^{-1}$]', fontsize=30)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()
