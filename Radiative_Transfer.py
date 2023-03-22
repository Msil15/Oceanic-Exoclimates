import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate, optimize
from hapi import *
import time
from decimal import *
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
g = 0.904*1e3 #TK CHANGED surface gravity (cm/s^2)
c1L = 3.7418e-5 #first radiation constant (erg cm^2/s)
c2 = 1.4387769 #second radiation constant (cm K)
bar2cgs = 1e6 #converts bar to cgs units
SB = 5.670367e-5 #Stefan-Boltzmann Constant (erg cm^-2 s^-2 K^-4)
R = 8.31e7 #Universal Gas Constant erg/K*mol

#INTENSITY FUNCTION
def Intensity(mu, L, tau, nu, T): #stellar luminosity, optical depth, angle of incoming intensity, wavenumber of incoming light, and temperature of star
    
    d = 0.7*1.5e13 #TK CHANGED distance between star and exoplanet -- 0.7 AU (cm)

    I_0 = L/(4*np.pi*(d)) #Total intensity(erg/s*cm)

    I = I_0*np.exp(tau/mu) #Beer's Law

    B = c1L*(nu**3)/(np.exp(c2*nu/T)-1) #Planck Function
    S = B #Source Function

    dI_dtau = (I - S)/mu
    I_integrated = (mu*I - S*tau)/mu
    return I

#DEFINE PRESSURE, COLUMN MASS, & HEIGHT ARRAYS
logp = np.linspace(-15,1.97,48) #logp = np.arange(-15,-12,0.01)
pressure = 10.0**logp #pressure in bars
bar2cgs = 1e6 #convert bar to cgs units
p0 = max(pressure) #BOA pressure
m = pressure*bar2cgs/g #column mass (g/cm^2)
m0 = p0*bar2cgs/g #BOA column mass

m_array = np.array(m)

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
N2 = Molecule(2*2.3259e-23, 28.0134, 0.95, 22, 1, 71163)
CH4 = Molecule(2.6569e-23, 16.04, 0.5, 6, 1, 71162)
CO2 = Molecule(7.3e-12, 44.01, 1, 2, 1, 0)

#CALCULATING OPTICAL DEPTH
def opacity_calc(t, p, molecule): #temp & pressure of atmosphere layers and bounds of integration
    
    nu = 20000 #Average wavenumber of solar radiation

    T_ref = 296
    P_ref = 1
    P_self = p
    
    '''
    Gamma_air = getColumn('H2O-O2_14000-15000', 'gamma_air')[molecule.nu] #air-broadened HWHM for T_ref & P_ref (cm^-1/atm)
    n_air = getColumn('H2O-O2_14000-15000', 'n_air')[molecule.nu] #coefficient of temperature dependence of gamma_air
    Gamma_self = getColumn('H2O-O2_14000-15000', 'gamma_self')[molecule.nu] #self-broadened HWHM for T_ref & P_ref (cm^-1/atm)
    pressure_shift = getColumn('H2O-O2_14000-15000', 'delta_air')[molecule.nu] #pressure shift of wavenumber for T_ref & P_ref (cm^-1/atm)
    elower = getColumn('H2O-O2_14000-15000', 'elower')[molecule.nu] #lower-state energy of transition (cm^-1)
    '''

    Gamma_air = getColumn('CO2_18000-19908', 'gamma_air')[molecule.nu] #air-broadened HWHM for T_ref & P_ref (cm^-1/atm)
    n_air = getColumn('CO2_18000-19908', 'n_air')[molecule.nu] #coefficient of temperature dependence of gamma_air
    Gamma_self = getColumn('CO2_18000-19908', 'gamma_self')[molecule.nu] #self-broadened HWHM for T_ref & P_ref (cm^-1/atm)
    pressure_shift = getColumn('CO2_18000-19908', 'delta_air')[molecule.nu] #pressure shift of wavenumber for T_ref & P_ref (cm^-1/atm)
    elower = getColumn('CO2_18000-19908', 'elower')[molecule.nu] #lower-state energy of transition (cm^-1)

    Gamma_L = ((T_ref/t)**n_air)*((Gamma_air*(p-P_self)) + (Gamma_self*P_self)) #Lorentz HWHM

    f_L = (1/np.pi)*((Gamma_L)/(Gamma_L**2 + (nu-(incoming + pressure_shift*p))**2)) #Lorentz profile

    #SPECTRAL LINE INTENSITY
    #S_ref = getColumn('H2O-O2_14000-15000', 'sw')[molecule.nu] #Spectral line intensity at 296K

    S_ref = getColumn('CO2_18000-19908', 'sw')[molecule.nu]

    Q = partitionSum(molecule.M,molecule.I,t)

    Q_ref = partitionSum(molecule.M,molecule.I,T_ref)

    term1 = Q_ref/Q
    term2 = (np.e**((-c2*elower/t)))/(np.e**((-c2*elower)/T_ref))
    term3 = (1 - (np.e**(-c2*incoming/t)))/(1 - (np.e**(-c2*incoming/T_ref)))
    S = S_ref*term1*term2*term3

    opacity = S*f_L
    #print('Opacity is ', opacity)
    return opacity

def tau_calc(t, p, m, molecule):
    nu_1, coef_1 = absorptionCoefficient_Voigt(SourceTables='CO2_18000', Environment ={'p':p/1.013,'T':t}, Diluent={'self':1}, HITRAN_units = False)
    nu_2, coef_2 = absorptionCoefficient_Voigt(SourceTables='CO2_18000', Environment ={'p':p/1.013,'T':t}, Diluent={'self':1}, HITRAN_units = True)
    u = m*(1/molecule.pmass) #column number density
    result = [coef_2[molecule.nu]*u, coef_1[molecule.nu]]
    return result


#ATMOSPHERE LAYERS (DEFINING INITIAL VALUES)
#incoming = getColumn('H2O-O2_14000-15000','nu')[H2O.nu] #transition wavenumber.
incoming = getColumn('CO2_18000', 'nu')[CO2.nu]
B_albedo = 0.75 #TK CHANGED Bond-Albedo (Earth-Value = 0.306, Europa Value = 0.67)

coef_list = []
tau_list = [0] #adding opacity of empty space (nothing)
intensity_list = []
del_tau_list = []
transmission_list = []
flux_list = []
net_flux_list = []

def T_calc(T_list, p_list, m):



    for T, p, M in zip(T_list[1:m-1], p_list[1:m-1], m_array): #calculating opacity, optical depth, and intensity.
        #opacity_H2O = opacity_calc(T, p, H2O)
        #opacity_O2 = opacity_calc(T, p, O2)
        #opacity_total = H2O.mfrac*opacity_H2O + O2.mfrac*opacity_O2
        #opacity_total = opacity_calc(T,p,CO2)
        #opacity_list.append(opacity_total)
        #tau = tau_calc(T, p, ab[0], ab[1], CO2)
        result = tau_calc(T, p, M, CO2)
        tau = result[0]
        coef = result[1]
        tau_list.append(tau)
        coef_list.append(coef)
        inten = Intensity(-1, 3.839e33, tau, incoming, 5777) #TK CHANGED
        intensity_list.append(inten)

    for i in range(0, m-2): #calculating delta tau and the transmission function.
        tau_ij = tau_list[i-1] - tau_list[i]
        del_tau_list.append(tau_ij)
        trans = float(2*mpmath.gammainc(-2,tau_ij)*(tau_ij**2))
        hyp_int = special.shichi(tau_ij)
        #trans = (tau_ij**2)*(hyp_int[0] - hyp_int[1]) - (tau_ij - 1)*np.exp(-tau_ij) #(this might be best approximation)
        #D = 2 #diffusicity factor (valid when del_tau << 1)
        #trans = np.exp(D*tau_ij)
        transmission_list.append(trans)

    #FLUX CALCULATIONS

    #Method obtained from Shapiro, Ralph, 1972
    
    Y_0 = B_albedo*intensity_list[-1]
    X_0 = intensity_list[-1] 

    flux_list.append(Y_0)

    for trans in transmission_list:
        Y_i = Y_0 - (1 - trans)*X_0/trans
        X_i = trans*X_0 + (1 - trans)*Y_i
        net_flux_list.append(Y_i)
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

def The_Method(T_list, m):

    K = np.zeros_like(T_list)
    K[0] = Ttop
    K[m - 1] = Tbot

    for i in range(1, m - 1): #creating list of known values (T_j^n)
        K[i] = 4*(T_list[i+1]**4 - T_list[i-1]**4)

        del_z = 2.5e7/(m-2) #2.012e7/(m-2) #cm
        rho = 93*0.001225 #7.30e-9 #g/cm^2
        A_ = 24.99735
        B = 55.18696
        C_ = -33.69137
        D = 7.948387
        E = -0.136638
        C_p = (A_ + B*(T_list[i]/1e3) + C_*(T_list[i]/1e3)**2 + D*((T_list[i]/1e3))**3 + E*(T_list[i]/1e3)**(-2))*227221.09 #1.148e7 #8.9e6 #erg/g*K
        #formula for C_p from Kitchin Research Group

        C = -SB/(2*rho*C_p*del_z)

    K_m = np.asmatrix(K)
    K_T = np.matrix.transpose(K_m)

    A = [[0 for m in range(m)]for n in range(m)] #generating matrix A from K values.

    for j in range(m):
        for i in range(m):
            try:
                A[i-1][i] = 16*(T_list[i + 1]**3)
            except:
                pass
            A[i][i] = -(1/C)
            A[i][i-1] = -16*(T_list[i - 2]**3)
        

    A[0][0] = 1
    for i in range(1, m):
        A[0][i] = 0

    A[m-1][m-1] = 1
    for i in range(0, m-1):
        A[m -1][i] = 0

    A_m = np.asmatrix(A)

    A_T = A_m**(-1) #np.linalg.inv(A_m)

    del_T = A_T*K_T #inverting matrix and solving for all del T_j^n+1 simultaneously.
    del_T_T = np.transpose(del_T)
    del_T_list = np.matrix.tolist(del_T_T)[0]

    for i in range(1, m-1):
        T_list[i] = T_list[i] + 1.9*del_T_list[i] #overrelaxation
    
    T_list[0] = T_list[1] #setting the bottom value for temp to be a function of the layer above it.
    return [T_list, del_T_list]

TOL_list = []

tic = time.perf_counter()

#pressure & temp values at top at bottom of atmosphere
Tbot = 740 #102
Ttop = 5.0

pbot = 93 #1e-12
ptop = 0

#creating evenly space list of temp and pressure between endpoints.
m = 50 #number of points (including endpoints)
T_list_0 = [*np.linspace(Ttop, Tbot, m)]
p_list_0 = [*np.linspace(ptop, pbot, m)]

for i in range(1000000):
    T_calc(T_list_0, p_list_0, m) #generate list of T & p values for current timestep
    Result = The_Method(T_list_0, m)
    T_list_new = Result[0]
    avg_list = []
    for delT, Tn in zip(Result[1][1:m-1], T_list_new[1:m-1]):
        rel_err = delT/Tn
        avg_list.append(rel_err)

    avg = sum(avg_list)/(m-2)
    print(avg)
    if abs(avg) <= 1e-5:
        TOL_list.append(avg)
                      
    if len(TOL_list) == 10: #if the change is below the given value for a certain number of loops, the tolerance has been reached
        print('Tolerance reached after', i, 'iterations')
        break

    coef_list = [] #resetting lists for next loop
    tau_list = [0]
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

toc = time.perf_counter()
print("Finished in ", toc - tic, "seconds.")

def poly_func(t, a, b, e, d):
    return a*(np.array(t))**3 + b*(np.array(t))**2 + e*(np.array(t)) + d

popt, pcov = optimize.curve_fit(poly_func, T_list_0[0:m-1], p_list_0[0:m-1])
plt.plot(T_list_0[0:m-1], p_list_0[0:m-1], 'ro-', label = 'data')
plt.plot(sorted(T_list_0[0:m-1]), poly_func(sorted(T_list_0[0:m-1]), *popt), 'b--', label = 'best fit')
plt.title('T-P Profile for Venus-like Atmosphere')
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (bar)')
#plt.ylim(10**(-11.9), 10**(-14.2))
plt.ylim(95, 0)
plt.yscale('log')
plt.legend()
plt.show()
plt.clf

print(transmission_list)

T_list_op, coef_list = zip(*sorted(zip(T_list_0[1:m-1], abs(coef_list))))
T_list_tau, tau_list = zip(*sorted(zip(T_list_0[1:m-1], abs(tau_list[0:m-2]))))

fig, axs = plt.subplots(2, 2, sharex=True)
fig.suptitle("Temperature (K) vs various quantities (Venus)")
axs[0,0].plot(T_list_op, coef_list)
axs[0,0].set_title("Absorption Coefficient ($cm^{-1}$)")
axs[0,0].set_yscale('log')
axs[0,1].plot(T_list_tau, tau_list)
axs[0,1].set_title("optical depth")
axs[0,1].set_yscale('log')
axs[1,0].plot(T_list_0[1:m-1], intensity_list)
axs[1,0].set_title("intensity ($erg/s \cdot cm$)")
axs[1,1].plot(T_list_0[1:m-1], net_flux_list)
axs[1,1].set_title('net flux ($erg/s \cdot cm$)')
plt.show()
plt.clf()

fig, axs = plt.subplots(2, 2, sharex=True)
fig.suptitle("Pressure (bar) vs various quantities (Venus)")
axs[0,0].plot(p_list_0[1:m-1], coef_list)
axs[0,0].set_title("Absorption Coefficient ($cm^{-1}$)")
axs[0,0].set_yscale('log')
axs[0,0].set_xscale('log')
axs[0,1].plot(p_list_0[1:m-1], tau_list[0:m-2])
axs[0,1].set_title("optical depth")
axs[0,1].set_yscale('log')
axs[0,1].set_xscale('log')
axs[1,0].plot(p_list_0[1:m-1], intensity_list)
axs[1,0].set_title("intensity ($erg/s \cdot cm$)")
axs[1,0].set_xscale('log')
axs[1,1].plot(p_list_0[1:m-1], net_flux_list)
axs[1,1].set_title('net flux ($erg/s \cdot cm$)')
axs[1,1].set_xscale('log')
plt.show()

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