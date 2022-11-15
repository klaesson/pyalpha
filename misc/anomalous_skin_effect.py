import numpy as np
from scipy import constants as con
import quadpy as qy

"""
Calculation of the anomalous skin effect for copper, as a function 
of temperature and magnetic field.

Valid in the temperature  range: 4.2 K < T < 900 K.
Valid in the magnetic field and RRR range (transverse magnetoresistance only):
10 < RRR*B[T] < 10000.

© Copyright[2020] CERN .This software is distributed under the 
terms of the GNU General Public Licence version 3 (GPL Version 3). In 
applying this licence, CERN does not waive the privileges and 
immunities granted to it by virtue of its status as an 
Intergovernmental Organization or submit itself to any jurisdiction.

Author: sergio.calatroni@cern.ch *)

References
[1] G.E.H. Reuter and E.H. Sondheimer, "The theory of the 
anomalous skin effect in metals", Proc. Roy. Soc. A195 (1948) 336-364 
[2] J.G. Hust and A.B. Lankford, "Thermal conductivity of aluminum, copper,
iron, and tungsten for temperatures from 1 K to the melting point",
NBSIR 84-3007, National Bureau of Standards (Boulder,Colorado, USA, 1984)
[3] E. Drexler, N. Simon and R. Reed, "Properties of copper and copper alloys 
at cryogenic temperatures," NIST Monograph 177,Chapter 8, US Government
Printing Office (Washington DC, USA, 1992)
[4] J.Bass, "Size effects", in Landolt-Börnstein, "Electrical resistivity of 
pure metals and dilute alloys" K.-H. Hellwege,J.L. Olsen (eds.), Vol 15A
"Electrical Resistivity, Kondo and Spin Fluctuation Systems, Spin Glasses and
Thermopower", Springer-Verlag (Berlin Heidelberg, DE, 1983)
[5] C. Kittel, "Introduction to solid state physics", 5th edition, Wiley 
(New York, USA, 1976)
[6] F. Caspers, M. Morvillo, F. Ruggiero and J. Tan, "Surface Resistance
Measurements and Estimate of the Beam-Induced Resistive Wall Heating of the 
LHC Dipole Beam Screen", LHC Project Report 307 (1999)
[7] W. Chou and F. Ruggiero, "Anomalous Skin Effect and Resistive Wall 
Heating", LHC Project Note 2 (1995)
[8] E. Métral and S. Calatroni, "Beam screen issues," Proc. HE-LHC Workshop,
pp.83\[Dash]89, 2010
[9] S. Calatroni, M. Arzeo, S. Aull, M. Himmerlich, P. Costa Pinto, 
W. Vollenberg, B. Di Girolamo, P. Cruikshank, P. Chiggiato, D. Bajek,
S. Wackerow and A. Abdolvand, "Cryogenic surface resistance of copper: 
Investigation of the impact of surface treatments for secondary electron yield
reduction", Phys. Rev. acc. Beams 22, 063101 (2019)
"""

# Constants and global quantities:
MU_0 = con.mu_0
EPSILON_0 = con.epsilon_0
ELECTRON_MASS = con.electron_mass
RHO_AMBIENT = 1.54e-8  # Phonon resistivity at 273 K of copper, from [2]
RHO_MFP = 6.6e-16  # Resistivity*(electron mean free path), Cu from [4]
V_F = 1.57e+6  # Fermi velocity of copper, from [5]


def rho(T_, RRR_):
    """ Calculation of copper resistivity, from [2] pages 8, 12, 22

    Valid for 30<RRR<3000.
    """
    P1 = 1.171e-17
    P2 = 4.49
    P3 = 3.841e+10
    P4 = -1.14
    P5 = 50
    P6 = 6.428
    P7 = 0.4531
    rho_0 = RHO_AMBIENT/(RRR_-1.)
    
    # If to avoid underflow in exp(-(P5/T_)**P6), exponent > ~500
    if T_ < 19:
        rho_i = P1*T_**P2
    else:
        rho_i = P1*T_**P2/(1. + P1*P3*T_**(P2 + P4)*np.exp(-(P5/T_)**P6))
    
    rho_i0 = P7*(rho_0*rho_i)/(rho_0 + rho_i)
                 
    return rho_0 + rho_i + rho_i0


def magneto_rho(T_, RRR_, B_):
    """ Calculation of copper mangetoresistivity, from [3] page 8-23

    Valid only for 10 < RRR_*B_[T] < 10000.
    """
    rho_tzero = rho(T_, RRR_)
    s_ = RHO_AMBIENT/rho_tzero
    
    if 10 <= s_*B_ <= 10000:
        formula = (10**(-2.662 + 0.3168*np.log10(B_*s_) 
                + 0.6229*np.log10(B_*s_)**2 - 0.1839*np.log10(B_*s_)**3 
                + 0.01827*np.log10(B_*s_)**4))
        return rho_tzero*(1 + formula)

    else:
        print('Func magneto_rho in anomalous_skin_effect file')
        print('Func indeterminate, input variabels are out of bounds')
        return


def effe_zero(alpha):
    """ Calculation of complex surface impedance in anomalous skin effect
    regime, from [1].
    
    
    The calculation is done for P = 1, ie specular reflection from surface
    """
    f_ = lambda t: (1./(t**2 + 1.j*alpha*np.real(1./(1.j*t)**3*(2.*1.j*t 
                    - (1. - (1.j*t)**2)*np.log((1. + 1.j*t)/(1. - 1.j*t))))))
    
    int_1, err_1 = qy.quad(f_, 0., 1.)
    int_2, err_2 = qy.quad(f_, 1., np.inf)
    return -2./np.pi*(int_1+int_2)


def zmks(rho, frequency):
    """ Valid only for copper
   
    Based on mfp=6.6*10^-16 / rho and vf=1.57*10^6 in MKS units.
    """
    mfp = RHO_MFP / rho
    omega = frequency*2.*np.pi
    delta = np.sqrt(2.*rho/MU_0/omega)
    alpha = 1.5*mfp**2/delta**2
    nelec = ELECTRON_MASS*V_F/(rho*(1.6e-19)**2*mfp)

    # Units below here are cgs
    vfCGS = V_F*100.
    massCGS = ELECTRON_MASS*1000.
    lightCGS = 2.99792458e+10
    chargeCGS = 4.803e-10
    nCGS = nelec/1e+6
    avalue = (np.sqrt(6)*(np.pi*omega/chargeCGS/lightCGS)**(2/3)*(massCGS
            *vfCGS/3./nCGS)**(1/3))
    ZCGS = -1.j*np.sqrt(8/3)*avalue*alpha**(1/3)*effe_zero(alpha)

    # Conversion to MKS 
    return ZCGS/np.sqrt(4.*np.pi*EPSILON_0)


# Other standard formulae, for example from [5]
def zs_normal(rho, frequency):
    """ Surface impedance in normal skin effect regime.
    
    """             
    return (1. + 1.j)*np.sqrt(np.pi*frequency*MU_0*rho)
                 

def pen_depth(rho, frequency):
    """ Skin depth.
    
    """
    return np.sqrt(2.*rho/(2.*np.pi*frequency*MU_0))
    

def r_infinity(frequency):
    """ Asymptotic limit of ASE
    
    Factor 8/9 is for specular reflection.
    """
    return 8./9.*(np.sqrt(3)/(16.*np.pi)*RHO_MFP
                *(2.*np.pi*frequency)**2*MU_0**2)**(1/3)


def rho_b(T_, RRR_, B_):
    """ Simplified magnetoresistivity formula used in the LHC context

    From [6].
    """
    return rho(T_, RRR_)*(1. + 10**(1.055*np.log10(B_*RRR_) - 2.69))


def rs_anomalous(rho, frequency):
    """ Simplified ASE formula from [7]
    
    alpha should be >= 3 to be in ASE regime and for the simplified
    model to be valid, same alpha as in ZMKS [1].
    """
    alpha = lambda rho, frequency: (3./2.*(RHO_MFP/rho
                                    /pen_depth(rho, frequency))**2)
    return r_infinity(frequency)*(1. + 1.157*alpha(rho, frequency)**(-0.276))





