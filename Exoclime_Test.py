#Initial code from Appendix E of "Exoplanetary Atmospheres" by Kevin Heng, 1st ed.
#This code creates a temperature-pressure profile for a planetary atmosphere
#and graphs a fiducial curve against modified curves.

import numpy as np
from scipy import special
from matplotlib import pyplot as plt

#function to compute T-P profile
def tpprofile(m,m0,tint,tirr,kappa_S,kappa0,beta_S0,beta_L0,el1,el3):
    albedo = (1.0-beta_S0)/(1.0+beta_S0)
    beta_S = kappa_S*m/beta_S0
    coeff1 = 0.25*(tint**4)
    coeff2 = 0.125*(tirr**4)*(1.0-albedo)
    term1 = 1.0/el1 + m*kappa0/el3/(beta_L0**2)
    term2 = 0.5/el1 + special.expn(2,beta_S)*(kappa_S/kappa0/beta_S0)
    term3 = kappa0*beta_S0*(1/3 - special.expn(4,beta_S))/el3/kappa_S/(beta_L0**2)
    result = (coeff1*term1 + coeff2*(term2 + term3))**0.25
    return result

#input parameters
g = 1e3 #surface gravity (cm/s^2)
tint = 150 #internal temperature (K)
tirr = 1200 #irradiation temperature (K)
kappa_S0 =  0.01 #shortwave opacity (cm^2/g)
kappa0 = 0.02 #infrared opacity (cm^2/g)
beta_S0 = 1.0 #shortwave scattering paramter
beta_L0 = 1.0 #longwave scattering parameter
el1 = 3/8 #first longwave Eddington coefficient
el3 = 1/3 #second longwave Eddington coefficient

#define pressure and column mass arrays
logp = np.arange(-3,2.01,0.01)
pressure = 10.0**logp #pressure in bars
bar2cgs = 1e6 #convert bar to cgs units
p0 = max(pressure) #BOA pressure
m = pressure*bar2cgs/g #column mass
m0 = p0*bar2cgs/g #BOA column mass

#Experiment 1: greenhouse effect
pn = len(m)
tp0 = np.zeros(pn)
tp1 = np.zeros(pn)
tp3 = np.zeros(pn)

for i in range(0, pn): #plotting a fiducial profile (all default parameters) against modified profiles.
    tp0[i] = tpprofile(m[i],m0,0.0,tirr,kappa_S0,kappa0,beta_S0,beta_L0,el1,el3) #fiducial profile
    tp1[i] = tpprofile(m[i],m0,0.0,tirr,kappa_S0,0.03,beta_S0,beta_L0,el1,el3) #profile with increased infrared opacity
    tp3[i] = tpprofile(m[i],m0,tint,tirr,kappa_S0,kappa0,beta_S0,beta_L0,el1,el3) #profile with internal temp 150 K

line1, = plt.plot(tp0, pressure, linewidth=2, color='k', linestyle='-')
line2, = plt.plot(tp1, pressure, linewidth =4, color ='k', linestyle ='--')
plt.plot(tp3, pressure, linewidth=2, color ='k', linestyle='-')
plt.yscale('log')
plt.xlim([800,1100])
plt.ylim([1e2,1e-3])
plt.xlabel('$T$ (K)', fontsize=18)
plt.ylabel('$P$ (bar)', fontsize=18)
plt.show()
