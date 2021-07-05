# -*- coding: utf-8 -*-
"""
Created on Tue Oct  20 21:59:45 2020

@author: Marc-Antoine

See 10.1103/RevModPhys.75.1
See http://savoirs.usherbrooke.ca/handle/11143/5054

Modified to implement use of physical parameters
"""

import numpy as np
import time


# Single dot functions
def U(N, Ng, Ec=1):
    """

    Parameters
    ----------
    N : Int
        Number of electrons on the dot.
    Ng : Float
        Dimensionless gate charge.
    Ec : Float, optional
        Charging energy. The default is 1.

    Returns
    -------
    Float
        Electrostatic energy of the single dot.

    """
    return 0.5*Ec*(N-Ng)**2

def Ng(Vg, Cg=1, e=1):
    """

    Parameters
    ----------
    Vg : Float
        Gate voltage.
    Cg : Float, optional
        Gate capacitance. The default is 1.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Dimensionless gate charge.

    """
    return e*Cg*Vg

def U_moy(Ng, N_max=10, kBT=0.01):
    """

    Parameters
    ----------
    Ng : Float
        Dimensionless gate charge.
    N_max : Int, optional
        Maximum number of electrons in a dot. The default is 10.
    kBT : Float, optional
        Thermal energy. The default is 0.01.

    Returns
    -------
    Float
        Average electrostatic energy of the single dot.

    """
    Z = 0 # partition function
    moy = 0
    for N in range(N_max+1): # loop from [0, N_max]
        E = U(N, Ng)
        Z = Z + np.exp(-E/kBT)

        moy = moy + E*np.exp(-E/kBT)

    return moy/Z

def N_moy(Ng, N_max=10, Ec=1, kBT=0.01):
    """

    Parameters
    ----------
    Ng : Float
        Dimensionless gate charge.
    N_max : Int, optional
        Maximum number of electrons in a dot. The default is 10.
    Ec : Float, optional
        Charging energy. The default is 1.
    kBT : Float, optional
        Thermal energy. The default is 0.01.

    Returns
    -------
    Float
        Average electron number in the single dot.

    """
    Z = 0
    moy = 0
    for N in range(N_max+1): # loop from [0, N_max]
        E = U(N, Ng, Ec=Ec)
        Z = Z + np.exp(-E/kBT)

        moy = moy + N*np.exp(-E/kBT)

    return moy/Z

###############################################################################

# Double dot functions
def f(N1, N2, Vg1, Vg2, Ec1, Ec2, Cg1, Cg2, Ecm, e):
    """

    Parameters
    ----------
    N1 : Int
        Number of electrons on dot 1.
    N2 : Int
        Number of electrons on dot 1.
    Vg1 : Float
        Voltage on gate 1.
    Vg2 : Float
        Voltage on gate 2.
    Ec1 : Float
        Charging energy of dot 1.
    Ec2 : Float
        Charging energy of dot 2.
    Cg1 : Float
        Capacitance of gate 1.
    Cg2 : Float
        Capacitance of gate 2.
    Ecm : Float
        Electrostatic coupling energy between the two dots.
    e : Float
        Elementary charge.

    Returns
    -------
    Float
        Electrostatic energy of charges on the gates.

    """
    return -1/e*(Cg1*Vg1*(N1*Ec1+N2*Ecm)+Cg2*Vg2*(N1*Ecm+N2*Ec2))\
        +1/e**2*(0.5*Cg1**2*Vg1**2*Ec1+0.5*Cg2**2*Vg2**2*Ec2+Cg1*Vg1*Cg2*Vg2*Ecm)


def U_DQD(N1, N2, Vg1, Vg2, Cg1, Cg2, Cm, CL, CR, e=1):
    """

    Parameters
    ----------
    N1 : Int
        Number of electrons on dot 1.
    N2 : Int
        Number of electrons on dot 2.
    Vg1 : Float
        Voltage on gate 1.
    Vg2 : Float
        Voltage on gate 2.
    Cg1 : Float
        Capacitance of gate 1.
    Cg2 : Float
        Capacitance of gate 2.
    Cm : Float
        Capacitance between the two dots.
    CL : Float
        Capacitance of the source.
    CR : Float
        Capacitance of the drain.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Electrostatic energy of the double dot.

    """
    C1 = CL+Cg1+Cm # sum of the capacitance attached to dot 1
    C2 = CR+Cg2+Cm
    Ec1 = e**2*(C2/(C1*C2-Cm**2)) #Ec1 = e**2/C1*(1/(1-(Cm**2/(C1*C2)))) # version from paper but diverges at Cm=0
    Ec2 = e**2*(C1/(C1*C2-Cm**2)) #Ec2 = e**2/C2*(1/(1-(Cm**2/(C1*C2))))
    Ecm = e**2*(Cm/(C1*C2-Cm**2)) #Ecm = e**2/Cm*(1/((C1*C2/Cm**2)-1))

    return 0.5*N1**2*Ec1+0.5*N2**2*Ec2+N1*N2*Ecm+f(N1, N2, Vg1, Vg2, Ec1, Ec2, Cg1, Cg2, Ecm, e=e)

# def N_moy_DQD(Vg1, Vg2, Cg1, Cg2, Cm, CL, CR, N_max = 10, kBT=0.01, e=1):
#     """
#
#     Parameters
#     ----------
#     Vg1 : Float
#         Voltage on gate 1.
#     Vg2 : Float
#         Voltage on gate 2.
#     Cg1 : Float
#         Capacitance of gate 1.
#     Cg2 : Float
#         Capacitance of gate 2.
#     Cm : Float
#         Capacitance between the two dots.
#     CL : Float
#         Capacitance of the source.
#     CR : Float
#         Capacitance of the drain.
#     N_max : Int, optional
#         Maximum number of electrons in a dot. The default is 10.
#     kBT : Float, optional
#         Thermal energy. The default is 0.01.
#     e : Float, optional
#         Elementary charge. The default is 1.
#
#     Returns
#     -------
#     Float
#         Average number of electrons in the double dot.
#
#     """
#     Z = 0 # partition function
#     moy = 0
#     for N2 in range(N_max+1): # loop from [0, N_max]
#         for N1 in range(N_max+1):
#             E = U_DQD(N1, N2, Vg1, Vg2, Cg1, Cg2, Cm, CL, CR, e=e)
#             Z = Z + np.exp(-E/kBT)
#
#             moy = moy + (N1+N2)*np.exp(-E/kBT)
#
#     return moy/Z


def N_moy_DQD(Vg1, Vg2, Cg1, Cg2, Cm, CL, CR, N_min=0, N_max=10, kBT=0.01, e=1, verbose=False):
    """

    Parameters
    ----------
    Vg1 : Float
        Voltage on gate 1.
    Vg2 : Float
        Voltage on gate 2.
    Cg1 : Float
        Capacitance of gate 1.
    Cg2 : Float
        Capacitance of gate 2.
    Cm : Float
        Capacitance between the two dots.
    CL : Float
        Capacitance of the source.
    CR : Float
        Capacitance of the drain.
    N_max : Int, optional
        Maximum number of electrons in a dot. The default is 10.
    kBT : Float, optional
        Thermal energy. The default is 0.01.
    e : Float, optional
        Elementary charge. The default is 1.

    Returns
    -------
    Float
        Average number of electrons in the double dot.

    """
    if verbose:
        number_loop = 0
        total_loop = N_max
        start = time.time()
        start_loop = 0

    Z = 0  # partition function
    moy = 0
    for N2 in range(N_min, N_max + 1):  # loop from [0, N_max]
        if verbose:
            end_loop = time.time()
            number_loop += 1
            if number_loop % 10 == 0:
                print(f'Loop number: {number_loop}\t'
                      f'Time left: {round((total_loop - number_loop) * (end_loop - start_loop), 2)}\t'
                      f'Time loop: {round(end_loop - start, 2)}')
            start_loop = time.time()

        for N1 in range(N_min, N_max + 1):
            E = U_DQD(N1, N2, Vg1, Vg2, Cg1, Cg2, Cm, CL, CR, e=e)
            Z = Z + np.exp(-E / kBT)

            moy = moy + (N1 - N2) * np.exp(-E / kBT)

    if verbose:
        end = time.time()
        print(f'Total time: {end - start}')
    return moy / Z

# Code examples
"""
###############################################################################
# Single quantum dot
###############################################################################
# Electrostatic energy without temperature
x = np.linspace(-1,1,101)
plt.figure()
U_array = []
for i in range(-1,3):
    y = U(i,x+i)
    U_array.append(y)
    plt.plot(x+i,y)


# Electrostatic energy with temperature
x = np.linspace(-0.5,3.1,1001)
y = U_moy(x)
plt.figure()
plt.plot(x,y)

# Charge transition at 2 temperatures and conductance peaks
plt.figure()
x = np.linspace(0,4,1001)
plt.plot(x,N_moy(x))
plt.plot(x,N_moy(x, kBT=0.1))
plt.plot(x,10*np.gradient(N_moy(x)))
"""

###############################################################################
# Double quantum dot
###############################################################################
"""
nx, ny = (500, 500)
x = np.linspace(0, 0.05, nx)
y = np.linspace(0, 0.05, ny)
xv, yv = np.meshgrid(x, y)
Cg = 10.3e-18
T = 0.1
kB = 1.381e-23
z = N_moy_DQD(xv, yv, Cg1=Cg, Cg2=Cg, Cm=0.4*Cg, CL=0.5*Cg, CR=0.5*Cg, N_max = 5, kBT = 2*kB*T, e=1.602e-19) #Physical Should not have factor 2 Eth

# z = N_moy_DQD(xv, yv, Cg1=0.7*2, Cg2=0.6*2, Cm=0.1*2, CL=0.2*2, CR=0.2*2, N_max = 5, kBT = 0.01, e=1)

# Stability diagram
plt.figure()
plt.pcolormesh(xv,yv,z,cmap="viridis",shading="auto")
plt.xlabel("Vg1 (V)")
plt.ylabel("Vg2 (V)")
cbar = plt.colorbar()
cbar.set_label("Nb of electrons", rotation=90)

# Derivative of the stability diagram
plt.figure()
plt.pcolormesh(xv,yv,np.gradient(z, axis=0),cmap="viridis",shading="auto")
plt.xlabel("Vg1 (V)")
plt.ylabel("Vg2 (V)")
plt.title(f'Stability diagram (' + r'$N_{max}=5$)' + f' @T={T}K')
cbar = plt.colorbar()
cbar.set_label("Conductance (S)", rotation=90)




# Trace in the stability diagram
plt.figure()
# index = round(len(z)/2)-30
index = np.where(y>=0)[0][0]
z2 = z[:,index]
plt.plot(x,z2, label=f"Vg2 = {y[index]:.3f} V")
plt.xlabel("Vg1 (V)")
plt.ylabel("Nb of electrons")
plt.legend()

plt.show()
"""

###############################################################################
# Parameters
###############################################################################

"""
UNSW quantum dots (voltage range for 5 electrons)
"""
# Cg = 10.3e-18 #UNSW capacitance
# Cg2 = Cg1
# CL = 5*Cg1
# CR = 5*Cg2
# nx, ny = (500, 500)
# x = np.linspace(0, 0.05, nx)
# y = np.linspace(0, 0.05, ny)
# xv, yv = np.meshgrid(x, y)


"""
QuTech quantum dots
"""
# Cg1 = 5.80e-18 #QuTech (https://www.nature.com/articles/s41586-021-03469-4)
# Cg2 = 4.56e-18 #QuTech
# CL = 2.5*Cg1
# CR = 2.7*Cg2
# nx, ny = (500, 500)
# x = np.linspace(0, 0.15, nx)
# y = np.linspace(0, 0.15, ny)
# xv, yv = np.meshgrid(x, y)


"""
Princeton quantum dots
"""
# Cg1 = 24.3e-18 # Petta's lab Princeton https://dataspace.princeton.edu/bitstream/88435/dsp01f4752k519/1/Zajac_princeton_0181D_12764.pdf
# Cg2 = Cg1
# CL = 0.08*Cg #Petta's lab
# CR = CL
# nx, ny = (500, 500)
# x = np.linspace(0, 0.035, nx)
# y = np.linspace(0, 0.035, ny)
# xv, yv = np.meshgrid(x, y)

"""
Sandia national lab
"""
# Cg1 = 1.87e-18 #Sandia National Lab https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7838537
# Cg2 = Cg1
# CL = 1.7*Cg
# CR = CL
# nx, ny = (500, 500)
# x = np.linspace(0, 0.4, nx)
# y = np.linspace(0, 0.4, ny)
# xv, yv = np.meshgrid(x, y)


"""
CEA LETI 
"""
# Cg2 = 19.7e-18 #CEA LETI Grenoble-Alpes https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.024066
# Cg1 = 10.3e-18 #CEA LETI
# CL = 0.1*Cg1
# CR = 0.2*Cg2
# nx, ny = (500, 500)
# x = np.linspace(0, 0.09, nx)
# y = np.linspace(0, 0.045, ny)
# xv, yv = np.meshgrid(x, y)


"""
UCL
"""
# Cg1 = 9.1e-19 #UCL (Single dot) https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.010353
# Cg2 = Cg1
# CL = 2.2*Cg
# CR = CL
# nx, ny = (500, 500)
# x = np.linspace(0, 0.95, nx)
# y = np.linspace(0, 0.95, ny)
# xv, yv = np.meshgrid(x, y)
