import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate
from hapi import *
import mpmath

#IMPORTING HITRAN DATA
db_begin('data')

#CONSTANTS
c = 2.99792458e10 #Speed of light (cm/s)
Na = 6.022e23 #Avagadro constant (mol^-1)
k = 1.308649e-16 #Boltzman constant (erg/K)
h = 6.6261e-27 #Planck's constant (cm^2*g/s)
g = 1e3 #surface gravity (cm/s^2)
c1L = 1.191e-5 #first radiation constant (erg cm^2/s)
c2 = 1.4387769 #second radiation constant (cm K)
bar2cgs = 1e6 #converts bar to cgs units
M = 18.010565 #molar mass of water vapor (g/mol)

#INTENSITY FUNCTION
def Intensity(L, tau, theta, nu, T): #stellar luminosity, optical depth, angle of incoming intensity, wavenumber of incoming light, and temperature of planet
    
    d = 1.5e13 #Distance between star and exoplanet -- 1 AU (cm)

    I_0 = L/(4*np.pi*d**2) #Total intensity(watts/cm)

    mu = np.cos(theta) #Angle of incoming intensity
    print('Mu is = ', mu)
    I = I_0*np.exp(tau/mu) #Beer's Law

    B = c1L*(nu**3)/(np.exp(c2*nu/T)-1) #Planck Function
    S = B #Source Function

    dI_dtau = (I - S)/mu
    print(S)
    return dI_dtau

#DEFINE PRESSURE, COLUMN MASS, & HEIGHT ARRAYS
logp = np.arange(-3,0,0.01)
pressure = 10.0**logp #pressure in bars
bar2cgs = 1e6 #convert bar to cgs units
p0 = max(pressure) #BOA pressure
m = pressure*bar2cgs/g #column mass (g/cm^2)
m0 = p0*bar2cgs/g #BOA column mass
height_list = []

for i in m:
    height = -3322.48*i + (3.24685e6)
    height_list.append(abs(height))

m_array = np.array(m)
height_array = np.array(height_list)


#CALCULATING OPTICAL DEPTH
def tau_calc(t, p, a, b):
    nu, opacity = absorptionCoefficient_Lorentz(SourceTables='H2O_Visible', Environment ={'T':t,'p':p})
    u = integrate.simps(height_array[a:b], x = m_array[a:b]) #column number density
    result = opacity[0]*u

    print(opacity[0], u)
    return result


#ATMOSPHERE LAYERS

incoming = getColumn('H2O_Visible','nu')[0] #wavenumber of incoming radiation

#layer 1
p1 = 1
t1 = 289.9
tau1 = tau_calc(t1, p1, 200, 299)
inten1 = Intensity(3.828e33, tau1, np.pi, incoming, 289.9)
print('optical depth is ', tau1, 'and intensity is ', inten1)
flux_up1 = 4
flux_down1 = 3


#layer 2
p2 = 0.0031
t2 = 227.70
tau2 = tau_calc(t2, p2, 100, 199)
inten2 = Intensity(3.828e33, tau2, np.pi, incoming, 289.9)
print('optical depth is ', tau2, 'and intensity is ', inten2)
flux_up1 = 4
flux_down1 = 3


#layer 3
p3 = 1e-3
t3 = 231.29
tau3 = tau_calc(t3, p3, 0, 99)
inten3 = Intensity(3.828e33, tau3, np.pi, incoming, 289.9)
print('optical depth is ', tau3, 'and intensity is ', inten3)
flux_up1 = 4
flux_down1 = 3

#TRANSMISSION FUNCTION
#tau2 > tau3 > tau1
tau21 = tau2 - tau1
tau13 = tau3 - tau1

print(tau1, tau2, tau3)

print(tau21)
print(tau13)

trans_21 = 2*mpmath.gammainc(-2.0 + 1e-15, tau21)*(tau21**2)
trans_13 = 2*mpmath.gammainc(-2.0 + 1e-15, tau13)*(tau13**2)

 #Transmission function returns ~1 as expected for low opacities, but atmosphere gets *more* opaque with higher altitude, not less.
print('')
print('%.16f ' % trans_21)
print('')
print('%.16f ' % trans_13)


#ITERATION
