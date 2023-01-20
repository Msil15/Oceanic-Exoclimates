import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate
from sympy import Symbol
from sympy import integrate as symintegrate
from hapi import *
import mpmath

#Might need to include scattering now that O2 is introduced?
#Partition function weighted according to Earth abundances
#Might need to include partial pressures

#IMPORTING HITRAN DATA
db_begin('data')

#CONSTANTS
c = 2.99792458e10 #Speed of light (cm/s)
Na = 6.022e23 #Avagadro constant (mol^-1)
k = 1.308649e-16 #Boltzman constant (erg/K)
h = 6.6261e-27 #Planck's constant (cm^2*g/s)
g = 0.134*1e3 #TK CHANGED surface gravity (cm/s^2)
c1L = 1.191e-5 #first radiation constant (erg cm^2/s)
c2 = 1.4387769 #second radiation constant (cm K)
bar2cgs = 1e6 #converts bar to cgs units
SB = 5.670367e-5 #Stefan-Boltzmann Constant (erg cm^-2 s^-2 K^-4)
R = 8.31e7 #Universal Gas Constant erg/K*mol

#INTENSITY FUNCTION
def Intensity(mu, L, tau, nu, T): #stellar luminosity, optical depth, angle of incoming intensity, wavenumber of incoming light, and temperature of planet
    
    d = 5.2*1.5e13 #TK CHANGED distance between star and exoplanet -- 5.2 AU (cm)

    I_0 = L/(4*np.pi*d**2) #Total intensity(watts/cm)

    I = I_0*np.exp(tau/mu) #Beer's Law

    B = c1L*(nu**3)/(np.exp(c2*nu/T)-1) #Planck Function
    S = B #Source Function

    dI_dtau = (I - S)/mu
    I_integrated = (mu*I - S*tau)/mu
    return I_integrated

#DEFINE PRESSURE, COLUMN MASS, & HEIGHT ARRAYS
logp = np.arange(-15,-12,0.01)
pressure = 10.0**logp #pressure in bars
bar2cgs = 1e6 #convert bar to cgs units
p0 = max(pressure) #BOA pressure
m = pressure*bar2cgs/g #column mass (g/cm^2)
m0 = p0*bar2cgs/g #BOA column mass
height_list = []

for i in m: #creating height array in terms of mass density
    height = -2.76e15*i + (2.012e7)
    height_list.append(abs(height))

m_array = np.array(m)
height_array = np.array(height_list)

#DEFINE MOLECULES
class Molecule:
    def __init__(self, pmass, mmass, mfrac, M, I, nu):
        self.pmass = pmass #mass of single particle (g)
        self.mmass = mmass #molar mass (g/mol)
        self.mfrac = mfrac #fraction of total atmosphere in terms of number of particles.
        self.M = M #HITRAN molecule number
        self.I = I #HITRAN isotopologue number
        self.nu = nu #row in HITRAN table to call values from.

H2O = Molecule(2.991e-23, 18.011, 0.1, 1, 1, 3570)
O2 = Molecule(5.3119e-23, 31.990, 0.99, 7, 1, 3571)

#CALCULATING OPTICAL DEPTH
def opacity_calc(t, p, molecule): #temp & pressure of atmosphere layers and bounds of integration
    
    nu = 20000 #Average wavenumber of solar radiation

    T_ref = 296
    P_ref = 1
    P_self = p
    
    Gamma_air = getColumn('H2O-O2_14000-15000', 'gamma_air')[molecule.nu] #air-broadened HWHM for T_ref & P_ref (cm^-1/atm)
    n_air = getColumn('H2O-O2_14000-15000', 'n_air')[molecule.nu] #coefficient of temperature dependence of gamma_air
    Gamma_self = getColumn('H2O-O2_14000-15000', 'gamma_self')[molecule.nu] #self-broadened HWHM for T_ref & P_ref (cm^-1/atm)
    pressure_shift = getColumn('H2O-O2_14000-15000', 'delta_air')[molecule.nu] #pressure shift of wavenumber for T_ref & P_ref (cm^-1/atm)
    elower = getColumn('H2O-O2_14000-15000', 'elower')[molecule.nu] #lower-state energy of transition (cm^-1)

    Gamma_L = ((T_ref/t)**n_air)*(Gamma_air*(p-P_self) + Gamma_self*P_self) #Lorentz HWHM

    f_L = (1/np.pi)*((Gamma_L)/(Gamma_L**2 + (nu-(incoming + pressure_shift*p))**2)) #Lorentz profile

    #SPECTRAL LINE INTENSITY
    S_ref = getColumn('H2O-O2_14000-15000', 'sw')[molecule.nu] #Spectral line intensity at 296K

    Q = partitionSum(molecule.M,molecule.I,t)

    Q_ref = partitionSum(molecule.M,molecule.I,T_ref)

    term1 = Q_ref/Q
    term2 = (np.e**((-c2*elower/t)))/(np.e**((-c2*elower)/T_ref))
    term3 = (1 - (np.e**(-c2*incoming/t)))/(1 - (np.e**(-c2*incoming/T_ref)))
    S = S_ref*term1*term2*term3

    opacity = S*f_L
    #print('Opacity is ', opacity)
    return opacity

def tau_calc(opacity, a, b):
    u = integrate.simps(height_array[a:b], x = m_array[a:b]) #column number density
    result = opacity*u
    return result


#ATMOSPHERE LAYERS (DEFINING INITIAL VALUES)
incoming = getColumn('H2O-O2_14000-15000','nu')[H2O.nu] #transition wavenumber.
B_albedo = 0.67 #TK CHANGED Bond-Albedo (Earth-Value = 0.306)

#pressure & temp values at top at bottom of atmosphere
Tbot = 102
Ttop = 5

pbot = 1e-12
ptop = 0

#creating evenly space list of temp and pressure between endpoints.
T_list_0 = [*np.linspace(Tbot, Ttop, 12)]
p_list_0 = [*np.linspace(pbot, ptop, 12)]

opacity_list = []
tau_list = []
intensity_list = []
del_tau_list = []
transmission_list = []
flux_list = []
net_flux_list = []

def T_calc(T_list, p_list, m):

    #bounds for calculating tau, so each optical depth is centered to its cell. 
    bound_list = [[269,284],[239,254],[209,214],[179,194],[149,164],[119,134],[89,104],[59,74],[29,44],[0,14]] 

    for T, p, ab in zip(T_list[1:m-1], p_list[1:m-1], bound_list): #calculating opacity, optical depth, and intensity.
        opacity_H2O = opacity_calc(T, p, H2O)
        opacity_O2 = opacity_calc(T, p, O2)
        opacity_total = H2O.mfrac*opacity_H2O + O2.mfrac*opacity_O2
        opacity_list.append(opacity_total)
        tau = tau_calc(opacity_total, ab[0], ab[1])
        tau_list.append(tau)
        inten = Intensity(-1, 3.828e33, tau, incoming, Tbot) #TK CHANGED
        intensity_list.append(inten)

    tau_list.append(0) #adding opacity of space (nothing)

    for i in range(0, m-2): #calculating delta tau and the transmission function.
        tau_ij = tau_list[i] - tau_list[i+1]
        del_tau_list.append(tau_ij)
        #trans = float(2*mpmath.gammainc(-2,tau_ij)*(tau_ij**2))
        #hyp_int = special.shichi(tau_ij)
        #trans = (tau_ij**2)*(hyp_int[0] - hyp_int[1]) - (tau_ij - 1)*np.exp(-tau_ij) #(this might be best approximation)
        D = 2 #diffusicity factor (valid when del_tau << 1)
        trans = np.exp(D*tau_ij)
        transmission_list.append(trans)

    #FLUX CALCULATIONS

    #Method obtained from Shapiro, Ralph, 1972

    Y_0 = B_albedo*intensity_list[-1]
    X_0 = intensity_list[-1] 

    flux_list.append([Y_0, X_0])

    for trans in transmission_list:
        Y_i = Y_0 - (1 - trans)*X_0/trans
        X_i = trans*X_0 + (1 - trans)*Y_i
        net_flux_list.append(Y_i - X_i)
        flux_list.append([Y_i, X_i])

        Y_0 = Y_i
        X_0 = X_i

    '''
    print(tau_list)
    print('')
    print(del_tau_list)
    print('')
    print(transmission_list)
    print('')
    print(flux_list)
    print('')
    print(net_flux_list)
    print('')
    '''
    return


#IMPLICIT ITERATION
rho = 7.30e-9 #g/cm^2
C_p = 2.8523*R*0.259825*1e7 #erg/g*K
del_z = 2012000 #cm

C = -SB/(2*rho*C_p*del_z)

def The_Method(T_list, m):

    K = np.zeros_like(T_list)
    K[0] = Tbot
    K[m - 1] = Ttop

    for i in range(1, m - 1): #creating list of known values (T_j^n)
        K[i] = 4*(T_list[i+1]**4 - T_list[i-1]**4)

    K_m = np.asmatrix(K)
    K_T = np.matrix.transpose(K_m)

    A = [[0 for m in range(m)]for n in range(m)] #generating matrix A from K values.

    for j in range(m):
        for i in range(m):
            try:
                A[i][i-1] = ((16*(T_list[i + 1]**3)))
            except:
                pass
            A[i][i] = ((1/C))
            A[i-1][i] = -((16*(T_list[i - 2]**3)))
        

    A[0][0] = 1 #float(K[0])*1.5
    for i in range(1, m):
        A[0][i] = 0

    A[m-1][m-1] = 1
    for i in range(0, m-1):
        A[m -1][i] = 0

    A_m = np.asmatrix(A)

    A_T = np.linalg.inv(A_m)

    del_T = A_T*K_T #inverting matrix and solving for all del T_j^n+1 simultaneously.
    del_T_T = np.transpose(del_T)
    del_T_list = np.matrix.tolist(del_T_T)[0]
    #print(del_T_list)

    for i in range(1, m-1):
        T_list[i] += del_T_list[i]
    
    T_list[0] = T_list[1]*1.1 #setting the bottom value for temp to be a function of the layer above it.
    print(T_list)
    return [T_list, del_T_list]

TOL_list = []

for i in range(1000):
    m = 12
    T_calc(T_list_0, p_list_0, m) #generate list of T & p values for current timestep
    Result = The_Method(T_list_0, m) 
    T_list_new = Result[0]
    del_T = sum(Result[1][1:m-1])/(m-2) #calculate average of del_T
    Tn_avg = sum(T_list_new[1:m-1])/(m-2)
    print(Result[1][1:m-1])
    if abs(del_T) <= 1e-5:
        TOL_list.append(del_T)
                      
    if len(TOL_list) == 100: #if the change is below the given value for a certain number of loops, the tolerance has been reached
        print('Tolerance reached after', i, 'iterations')
        break

    opacity_list = [] #resetting loops for next loop
    tau_list = []
    intensity_list = []
    del_tau_list = []
    transmission_list = []
    flux_list = []
    net_flux_list = []

    print('')
    print('')
    print('')
    for p, Tn, Tn1 in zip(p_list_0, T_list_0, T_list_new): #creating new pressure and temperature arrays
        p = (p/Tn)*Tn1 #Gay-Lussac's law
    T_list_0 = T_list_new

#SINKS & ENERGY BUDGETS

#ITERATION (OLD)
def Planck(nu, T):
    B = c1L*(nu**3)/(np.e**(c2*nu/T)-1)
    return B

'''
    J_up = 0.5*integrate.quad(Intensity, 0, 1, args =(3.828e33, tau, incoming, 102))[0]
    J_down = 0.5*integrate.quad(Intensity, -1, 0, args =(3.828e33, tau, incoming, 102))[0]


    expr = Tn1 - ((opacity_total)/Cp)*(J_up + J_down - 4*np.pi*Planck(incoming, Tn1)) - Tn
    #expr = del_T + Tn - ((O2.mfrac*opacity_calc(del_T + Tn, Pn1, O2, 'p') + H2O.mfrac*opacity_calc(del_T + Tn, Pn1, H2O, 'p'))/Cp)*(J_up + J_down - 4*np.pi*Planck(incoming, del_T + Tn)) - Tn #Figure out TIPS
    return expr
'''

#T, p --> opacity & optical depth
#optical depth --> intensity & transmission func
#trans func --> flux
#net flux --> update T, p
#round and round we go

#possible expansions; 
#realistic atmosphere comp
#clouds
#ocean source & sink (spectral energy budget)

