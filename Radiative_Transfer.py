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
    #print('Mu is = ', mu)
    I = I_0*np.exp(tau/mu) #Beer's Law

    B = c1L*(nu**3)/(np.exp(c2*nu/T)-1) #Planck Function
    S = B #Source Function

    dI_dtau = (I - S)/mu
    #print(S)
    return dI_dtau

#DEFINE PRESSURE, COLUMN MASS, & HEIGHT ARRAYS
logp = np.arange(-3,0,0.01)
pressure = 10.0**logp #pressure in bars
bar2cgs = 1e6 #convert bar to cgs units
p0 = max(pressure) #BOA pressure
m = pressure*bar2cgs/g #column mass (g/cm^2)
m0 = p0*bar2cgs/g #BOA column mass
height_list = []

for i in m: #creating height array in terms of mass density
    height = -3322.48*i + (3.24685e6)
    height_list.append(abs(height))

m_array = np.array(m)
height_array = np.array(height_list)

#1) use gray opacity to check if function is correct (done)
#2) make sure integration is from top down (done)
#3) expand to more layers for more granularity.
#4) use Exoclime_Test to create T-P profile for more accurate T-P inputs.

#CALCULATING OPTICAL DEPTH
def tau_calc(t, p, a, b): #temp & pressure of atmosphere layers and bounds of integration
    
    nu = 20000 #Average wavenumber of solar radiation

    T_ref = 296
    P_ref = 1
    P_self = p
    
    Gamma_air = getColumn('H2O_Visible', 'gamma_air')[0] #air-broadened HWHM for T_ref & P_ref (cm^-1/atm)
    n_air = getColumn('H2O_Visible', 'n_air')[0] #coefficient of temperature dependence of gamma_air
    Gamma_self = getColumn('H2O_Visible', 'gamma_self')[0] #self-broadened HWHM for T_ref & P_ref (cm^-1/atm)
    pressure_shift = getColumn('H2O_Visible', 'delta_air')[0] #pressure shift of wavenumber for T_ref & P_ref (cm^-1/atm)
    elower = getColumn('H2O_Visible', 'elower')[0] #lower-state energy of transition (cm^-1)

    Gamma_L = ((T_ref/t)**n_air)*(Gamma_air*(p-P_self) + Gamma_self*P_self) #Lorentz HWHM

    f_L = (1/np.pi)*((Gamma_L)/(Gamma_L**2 + (nu-(incoming + pressure_shift*p))**2)) #Lorentz profile

    #SPECTRAL LINE INTENSITY
    S_ref = getColumn('H2O_Visible', 'sw')[0] #Spectral line intensity at 296K

    Q = partitionSum(1,1,t)
    Q_ref = partitionSum(1,1,T_ref)

    term1 = Q_ref/Q
    term2 = np.exp((-c2*elower/t))/np.exp((-c2*elower)/T_ref)
    term3 = (1 - np.exp(-c2*incoming/t))/(1 - np.exp(-c2*incoming/T_ref))
    S = S_ref*term1*term2*term3

    opacity = S*f_L

    #nu, opacity = absorptionCoefficient_Lorentz(SourceTables='H2O_Visible', Environment ={'T':t,'p':p})
    u = integrate.simps(height_array[a:b], x = m_array[a:b]) #column number density
    result = opacity*u

    print('Opacity is ', opacity, ', and u is ', u)
    return result


#ATMOSPHERE LAYERS (DEFINING INITIAL VALUES)

incoming = getColumn('H2O_Visible','nu')[0] #transition wavenumber.
B_albedo = 0.306 #Bond-Albedo (Earth-Value = 0.306)

#layer 1
p1 = 1
t1 = 289.9

#layer 2
p2 = 0.0031
t2 = 231.29

#layer 3
p3 = 1e-3
t3 = 227.70

#layer 4
p4 = 0
t4 = 2.7

tau1 = tau_calc(t1, p1, 200, 299)
inten1 = Intensity(3.828e33, tau1, np.pi, incoming, 289.9)
print('optical depth is ', tau1, 'and intensity is ', inten1)

tau2 = tau_calc(t2, p2, 100, 199)
inten2 = Intensity(3.828e33, tau2, np.pi, incoming, 289.9)
print('optical depth is ', tau2, 'and intensity is ', inten2)

tau3 = tau_calc(t3, p3, 0, 99)
inten3 = Intensity(3.828e33, tau3, np.pi, incoming, 289.9)
print('optical depth is ', tau3, 'and intensity is ', inten3)

tau4 = tau_calc(t4, p4, 0, 99)
inten4 = Intensity(3.828e33, tau4, np.pi, incoming, 289.9)
print('optical depth is ', tau4, 'and intensity is ', inten4)

tau_G = 1 #optical depth of ground

tau12 = tau1 - tau2
tau23 = tau2 - tau3
tau34 = tau3 - tau4

print(tau1, tau2, tau3, tau4)

print(tau12)
print(tau23)
print(tau34)

trans_12 = 2*mpmath.gammainc(-2.0 + 1e-15, tau12)*(tau12**2)
trans_23 = 2*mpmath.gammainc(-2.0 + 1e-15, tau23)*(tau23**2)
trans_34 = 2*mpmath.gammainc(-2.0 + 1e-15, tau34)*(tau34**2)

print('')
print('%.16f ' % trans_12)
print('')
print('%.16f ' % trans_23)
print('')
print('%.16f ' % trans_34)
print('')

#FLUX CALCULATIONS(INITIAL VALUES)
def Planck(nu, T):
    B = c1L*(nu**3)/(np.exp(c2*nu/T)-1)
    return B

'''
OLD HENG FLUX
F_up_1 = np.pi*inten1 - B_albedo*inten1
F_up_2 = F_up_1*trans_12 + np.pi*Planck(incoming, t2)*(1 - trans_12) 
F_up_3 = F_up_2*trans_23 + np.pi*Planck(incoming, t3)*(1 - trans_23)

F_down_3 = np.pi*inten3
F_down_2 = F_down_3*trans_23 + np.pi*Planck(incoming, t2)*(1 - trans_23)
F_down_1 = F_down_2*trans_12 + np.pi*Planck(incoming, t1)*(1 - trans_12)
'''

#Method obtained from Shapiro, Ralph, 1972

Y_0 = B_albedo*inten4
X_0 = inten4

Y_1 = (Y_0 - (1 - trans_34)*X_0)/trans_34
X_1 = trans_34*X_0 + (1- trans_34)*Y_1

Y_2 = (Y_1 - (1 - trans_23)*X_1)/trans_23
X_2 = trans_23*X_1 + (1- trans_34)*Y_2

Y_3 = (Y_2 - (1 - trans_12)*X_2)/trans_12
X_3 = trans_12*X_2 + (1- trans_12)*Y_3

net_flux = Y_0 - X_0
net_flux1 = Y_1 - X_1
net_flux2 = Y_2 - X_2
net_flux3 = Y_3 - X_3

print(net_flux1)
print(net_flux2)
print(net_flux3)
print(X_0, X_1, X_2, X_3)
print(Y_0, Y_1, Y_2, Y_3)

#net flux in, temp & pressure go up, vice versa; that changes flux, and on and on

#SINKS & ENERGY BUDGETS

#ITERATION

#relaxation method
#define relative and abs tolerance
#Need to solve PDE

#1D mesh, y/j is altitude, each cell has temp and pressure associated with it.

#Define values at boundaries (ground, edge of space)

#Choose arbitrary initial values for mesh
#Sweep across mesh and update all points

#T, p --> opacity & optical depth
#optical depth --> intensity & transmission func
#trans func --> flux
#net flux --> update T, p
#round and round we go

#possible expansions; 
#realistic atmosphere comp
#clouds
#ocean source & sink (spectral energy budget)

