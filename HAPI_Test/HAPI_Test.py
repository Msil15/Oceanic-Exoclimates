import numpy
import matplotlib.pyplot as plt
from hapi import *

'''
-----------------------------------------
H2O summary:
-----------------------------------------
Number of rows: 115273
Table type: column-fixed
-----------------------------------------
            PAR_NAME           PAR_FORMAT

            molec_id                  %2d
        local_iso_id                  %1d
                  nu               %12.6f
                  sw               %10.3E
                   a               %10.3E
           gamma_air                %5.4f
          gamma_self                %5.3f
              elower               %10.4f
               n_air                %4.2f
           delta_air                %8.6f
 global_upper_quanta                 %15s
 global_lower_quanta                 %15s
  local_upper_quanta                 %15s
  local_lower_quanta                 %15s
                ierr                  %6s
                iref                 %12s
    line_mixing_flag                  %1s
                  gp                %7.1f
                 gpp                %7.1f
-----------------------------------------
Help on function PROFILE_VOIGT in module hapi:

PROFILE_VOIGT(Nu, GammaD, Gamma0, Delta0, WnGrid, YRosen=0.0, Sw=1.0)
    # Voigt profile based on HTP.
    # Input parameters:
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      GammaD    : Doppler HWHM in cm-1 (Input)
    #      Gamma0    : Speed-averaged line-width in cm-1 (Input).
    #      Delta0    : Speed-averaged line-shift in cm-1 (Input).
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    #      YRosen    : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)
'''


db_begin('data')

Nu = 1112800
GammaD = 0.0501
Gamma0 = 0.281
Delta0 = -0.011714
Dw = 1.
Wn = np.arange(Nu-Dw, Nu+Dw, 0.01) #Grid with step 0.01

PROF = PROFILE_VOIGT(Nu, GammaD, Gamma0, Delta0, Wn)[0]

print(PROF)