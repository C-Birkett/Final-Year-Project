# This is the main file for this project

import numpy as np
from scipy import linalg as scipyLinalg
import cmath
import math
import matplotlib.pyplot as plt

# --- Constants ---

# First define the fitting parameters for the effective TBM Hamiltonian of monolayer NbSe2, energy parameter are in eV

epsilon_1 = 1.4466
epsilon_2 = 1.8496

t_0 = -0.2308
t_1 = 0.3116
t_2 = 0.3459
t_11 = 0.2795
t_12 = 0.2787
t_22 = -0.0539

r_0 = 0.0037
r_1 = -0.0997
r_2 = 0.0385
r_11 = 0.0320
r_12 = 0.0986

u_0 = 0.0685
u_1 = -0.0381
u_2 = 0.0535
u_11 = 0.0601
u_12 = -0.0179
u_22 = -0.0425

lambda_SOC = 0.0784

PLC = 3.45e-10

# define alpha and beta from the wavevector k

def alpha(k):
    return 1/2*k[0]*PLC

def beta(k):
    return (math.sqrt(3)/2)*k[1]*PLC

# --- Potential 'jumps' ---

# define the 'hops' for each state (in order d_(z^2), d(x^2-y^2), d_(xy))

def V_0(k):
    tmp = epsilon_1
    tmp += 2*t_0*(2*math.cos(alpha(k))*math.cos(beta(k))+math.cos(2*alpha(k)))
    tmp += 2*r_0*(2*math.cos(3*alpha(k))*math.cos(beta(k))+math.cos(2*beta(k)))
    tmp += 2*u_0*(2*math.cos(2*alpha(k))*math.cos(2*beta(k))+math.cos(4*alpha(k)))
    return tmp

def Re_V_1(k):
    tmp = -2*math.sqrt(3)*t_2*math.sin(alpha(k))*math.sin(beta(k))
    tmp += 2*(r_1+r_2)*math.sin(3*alpha(k))*math.sin(beta(k))
    tmp += -2*math.sqrt(3)*u_2*math.sin(2*alpha(k))*math.sin(2*beta(k)) 
    return tmp

def Im_V_1(k):
    tmp = 2*t_1*math.sin(alpha(k))*(2*math.cos(alpha(k))+math.cos(beta(k)))
    tmp += 2*(r_1-r_2)*math.sin(3*alpha(k))*math.cos(beta(k))
    tmp += 2*u_1*math.sin(2*alpha(k))*(2*math.cos(2*alpha(k))+math.cos(2*beta(k)))
    return tmp

def V_1(k):
    return complex(Re_V_1(k), Im_V_1(k))

def Re_V_2(k):
    tmp = 2*t_2*(math.cos(2*alpha(k))-math.cos(alpha(k))*math.cos(beta(k)))
    tmp += (-2/math.sqrt(3))*(r_1 + r_2)*(math.cos(3*alpha(k))*math.cos(beta(k))-math.cos(2*beta(k)))
    tmp += 2*u_2*(math.cos(4*alpha(k))-math.cos(2*alpha(k))*math.cos(2*beta(k)))
    return tmp

def Im_V_2(k):
    tmp = 2*math.sqrt(3)*t_1*math.cos(alpha(k))*math.sin(beta(k))
    tmp += (2/math.sqrt(3))*(r_1-r_2)*math.sin(beta(k))*(math.cos(3*alpha(k))+2*math.cos(beta(k)))
    tmp += 2*math.sqrt(3)*u_1*math.cos(2*alpha(k))*math.sin(2*beta(k))
    return tmp

def V_2(k):
    return complex(Re_V_2(k), Im_V_2(k))

def V_11(k):
    tmp = epsilon_2
    tmp += (t_11 + 3*t_22)*math.cos(alpha(k))*math.cos(beta(k))
    tmp += 2*t_11*math.cos(2*alpha(k))
    tmp += 4*r_11*math.cos(3*alpha(k))*math.cos(beta(k))
    tmp += 2*(r_11+math.sqrt(3)*r_12)*math.cos(2*beta(k))
    tmp += (u_11+3*u_22)*math.cos(2*alpha(k))*math.cos(2*beta(k))
    tmp += 2*u_11*math.cos(4*alpha(k))
    return tmp

def Re_V_12(k):
    tmp = math.sqrt(3)*(t_22-t_11)*math.sin(alpha(k))*math.sin(beta(k))
    tmp += 4*r_12*math.sin(3*alpha(k))*math.sin(beta(k))
    tmp += math.sqrt(3)*(u_22-u_11)*math.sin(2*alpha(k))*math.sin(2*beta(k))
    return tmp

def Im_V_12(k):
    tmp = 4*t_12*math.sin(alpha(k))*(math.cos(alpha(k))-math.cos(beta(k)))
    tmp += 4*u_12*math.sin(2*alpha(k))*(math.cos(2*alpha(k))-math.cos(2*beta(k)))
    return tmp

def V_12(k):
    return complex(Re_V_12(k), Im_V_12(k))

def V_22(k) :
    tmp = epsilon_2
    tmp += ((3*t_11)+t_22)*math.cos(alpha(k))*math.cos(beta(k))
    tmp += 2*t_22*math.cos(2*alpha(k))
    tmp += 2*r_11*((2*math.cos(3*alpha(k))*math.cos(beta(k)))+math.cos(2*beta(k)))
    tmp += (2/math.sqrt(3))*r_12*((4*math.cos(3*alpha(k))*math.cos(beta(k)))-math.cos(2*beta(k)))
    tmp += ((3*u_11)+u_22)*math.cos(2*alpha(k))*math.cos(2*beta(k))
    tmp += 2*u_22*math.cos(4*alpha(k))
    return tmp

# --- Components of Hamiltonian ---

# Next we define H_TNN the matrix of nearest neighbors

def Hamiltonian_nearest_neighbors(k):
    return np.array([[V_0(k), V_1(k), V_2(k)],
                    [V_1(k).conjugate(), V_11(k), V_12(k)],
                    [V_2(k).conjugate(), V_12(k).conjugate(), V_22(k)]])

#print(type(V_1(k)))

# L_z which describes the difference in energy due to spin orbit coupling

L_z = np.array([[0,0,0],
                [0,0,complex(0,-2)],
                [0,complex(0,2),0]])

# The pauli matrices sigma_0 and sigma_z

sigma_0 = np.array([[1,0],
                    [0,1]])

sigma_z = np.array([[1,0],
                    [0,-1]])

test_array = np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]])

#print(np.kron(sigma_0,test_array))

# --- Hamiltonian ---

def Hamiltonian(k):
    return np.kron(sigma_0, Hamiltonian_nearest_neighbors(k)) + np.kron(sigma_z, (1/2)*lambda_SOC*L_z)

#print(Hamiltonian([1,1]))

#TODO define some sort of eigenvector, do across range of k values that are on paths K M, Gamma etc

#eValues, eVectors = np.linalg.eig(Hamiltonian)

#print("\n", eValues)
#print("\n", eVectors)

# --- Vectors ---

# Lengths:
# gamma -> m = 1/2 b (b = reciprocal lattice constant)
# gamma -> k = b/sqrt(3)
# k -> m = b/2*sqrt(3)

# reciprocal lattice constant
RLC = (4*np.pi)/(np.sqrt(3)*PLC)

# 2D rotation matrix
def rotation_matrix(theta):
    return np.array([[np.cos(theta), - np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

#define some with exact values to test
def rotation_matrix_90(theta):
    return np.array([[0, -1],
                    [1, 0]])


# A simple vector class to cut down on the amount of messy numpy array syntax & to keep methods such as rotation etc in one place
# these vectors are dimensionless, each element is assumed to be a function of some other value (i.e wavevector k)
class Vector:
    def __init__(this, vec: np.array):
        this.v = np.array(vec)
    
    def rotate(this, theta):
        this.v = np.matmul(rotation_matrix(theta), this.v)
        return this.v

    def rotate90(this, theta):
        this.v = np.matmul(rotation_matrix_90(theta), this.v)
        return this.v

    def len(this):
        return np.linalg.norm(this.v)

    def k(this, k: np.array):   # evaluate at some wavevector k, elementwise product
        return np.array([this.v[0] * k[0], this.v[1] * k[1]])

"""
# testing out the vector class
test_vector_1 = Vector([1,0])
test_vector_2 = Vector([0,1])
print ("\n init test vectors: ", test_vector_1.v, test_vector_2.v)

print("\n types: ", type(test_vector_1), type(test_vector_1.v))

test_vector_1.rotate(np.pi/2)
test_vector_2.rotate(np.pi/2)
print ("\n rotated test vectors: ", test_vector_1.v, test_vector_2.v)

print ("\n evaluate test vectors at [0.5,0.5]: ", test_vector_1.k([0.5,0.5]), test_vector_2.k([0.5,0.5]))
"""


""" backup of old lattice vectors (as functions rather than Vector class)
# Primitive lattice vectors
def PLV_1(k: np.array):
    return np.array([np.sqrt(3)*PLC*k[0]/2, PLC*k[1]/2])

def PLV_2(k: np.array):
    return np.array([-np.sqrt(3)*PLC*k[0]/2, PLC*k[1]/2])

# Reciprocal lattice vectors
def RLV_1(k: np.array):
    return np.array([2*np.pi*k[0]/PLC, 2*np.pi*k[1]/(np.sqrt(3)*PLC)])

def RLV_2(k: np.array):
    return np.array([2*np.pi*k[0]/PLC, -2*np.pi*k[1]/(np.sqrt(3)*PLC)])

print("PLC = ", PLC, "\nRLC = ", RLC)
print("|PLV| = ", np.linalg.norm(PLV_1([1,1])), "\n|RLV| = ", np.linalg.norm(RLV_1([1,1])))
"""

# TODO - make a lattice class, then can do lattice.generate_vectors(), lattice.rotate_vectors() etc
# can even do lattice.find_eigenvalues(path) etc
# this will make layering and rotating lattices easier. Can even extend functionality to superlattices.
# for now this will be useful for adding the 2nd layer and then rotating it

# for simplicity we will define the "default" primitive and reciprocal lattice vectors the same way they were defined before (before weekend of 13/11/21), this can be easily changed in the future if need be

# Primitive lattice vectors
PLV_1 = Vector([np.sqrt(3)*PLC/2, PLC/2])
PLV_2 = Vector([-np.sqrt(3)*PLC/2, PLC/2])

# Reciprocal lattice vectors
RLV_1 = Vector([2*np.pi/PLC, 2*np.pi/(np.sqrt(3)*PLC)])
RLV_2 = Vector([2*np.pi/PLC, -2*np.pi/(np.sqrt(3)*PLC)])

# Rotate the reciprocal lattice here:
RLV_1.rotate90(0)
RLV_2.rotate90(0)

# testing with weird angles
#RLV_1.rotate(5*np.pi/7)
#RLV_2.rotate(5*np.pi/7)


# Next we define the vectors that describe the paths for the brillouin zone as functions of k

Gamma_to_M = Vector((1.0/2.0)*RLV_1.v)

Gamma_to_K_prime = Vector((1.0/3.0) * (RLV_1.v + RLV_2.v))

M_to_K_prime = Vector((-1.0/6.0)*RLV_1.v + (1.0/3.0)*RLV_2.v)

M_to_K = Vector((1.0/6.0)*RLV_1.v - (1.0/3.0)*RLV_2.v)

#to avoid rotating twice
K_prime_to_M = Vector(M_to_K.v)

Gamma_to_K = Vector((2.0/3.0)*RLV_1.v - (1.0/3.0)*RLV_2.v)

K_to_Gamma = Vector((-2.0/3.0)*RLV_1.v + (1.0/3.0)*RLV_2.v)

M_to_M = [M_to_K, K_to_Gamma, Gamma_to_K_prime, K_prime_to_M]

"""
# test these make a closed loop - returns 10-7 error in x, which is negligible
tmp = np.array([0.0,0.0])
print("\ntmp = ", tmp)
for v in M_to_M:
    tmp += v.v
    print("\nvector = ", v.v)
    print("\ntmp = ", tmp)

print("this should be equal to zero: ", tmp)
"""

# conditions ai dot bj = 2Pi delta(i,j)
# pairwise parallel or perpendicular

# define the reciprocal lattice constant

# define better transform from real to reciprocal

# find coordinates of M, K, K'


def Figure_d():

    # this is not great but it doesn't work otherwise (numpy)
    Energy_1 = np.array([])
    Energy_2 = np.array([])
    Energy_3 = np.array([])
    Energy_4 = np.array([])
    Energy_5 = np.array([])
    Energy_6 = np.array([])
    Energy = [Energy_1,Energy_2,Energy_3,Energy_4,Energy_5,Energy_6]
    
    EVec1 = np.array([[]])
    EVec2 = np.array([[]])
    EVec3 = np.array([[]])
    EVec4 = np.array([[]])
    EVec5 = np.array([[]])
    EVec6 = np.array([[]])
    EigenVectors = [EVec1, EVec2, EVec3, EVec4, EVec5, EVec6]

    Path_x = np.array([])
    Path_y = np.array([])
    Path = np.array([])

    Path_Offset = 0 #for plotting x axis
    #assuming starting from point M - should rotate with the rest
    k_last = Gamma_to_M.v
    #k_last = Gamma_to_M.v
    print("\nstarting k offset (M) = ", k_last/RLC)
    Path_Offset += np.linalg.norm(k_last)
    print("\ninitial x offset = ", Path_Offset)

    for vectors in M_to_M:
        print("\nmoving along vector: ", vectors.v)
        for x in np.arange(0, 1, 0.01):

            k_step = x*vectors.v + k_last

            # for plotting the path in the brillouin zone
            Path_x = np.append(Path_x, k_step[0]/RLC)
            Path_y = np.append(Path_y, k_step[1]/RLC)

            Path = np.append(Path, x*vectors.len() + Path_Offset)
            #eValues, eVectors = np.linalg.eig(Hamiltonian(k_step)) #evalues and evectors (slow)
            eValues = np.linalg.eigvalsh(Hamiltonian(k_step))   #just evalues
            eValues.sort()
            
            for i in np.arange(0, 6, 1):
                    Energy[i] = np.append(Energy[i], eValues[i].real)
                    #EigenVectors[i] = np.append(EigenVectors[i], eVectors[i], axis = 1)

        k_last += vectors.v
        print("\nnew k offset = ", k_last/RLC)

        Path_Offset += vectors.len()
        print("new x offset = ", Path_Offset)

    return Energy, Path, EigenVectors, Path_x, Path_y

plot_y, plot_x, eigvec, path_x, path_y = Figure_d()

# plot the path in k that is taken
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)

ax.plot(path_x,
        path_y,
        marker = 'o',
        color = 'black',
        markersize = 2)


ax.plot(0,0, marker = 'o', color = 'red')
ax.plot(Gamma_to_M.v[0]/RLC,Gamma_to_M.v[1]/RLC, marker = 'o', color = 'red')
ax.plot(Gamma_to_K.v[0]/RLC,Gamma_to_K.v[1]/RLC, marker = 'o', color = 'red')
ax.plot(Gamma_to_K_prime.v[0]/RLC,Gamma_to_K_prime.v[1]/RLC, marker = 'o', color = 'red')

plt.axhline(y=0, xmin=-2, xmax=2, color = 'green')
plt.axvline(x=0, ymin=-2, ymax=2, color = 'green')

plt.xlim(-2,2)
plt.ylim(-2,2)

plt.show()


# plot the eigenvalues (electronic bands)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)

for i in np.arange(0,6,1):
    if (i % 2) == 0:
        colour = 'red'
    else:
        colour = 'blue'

    ax.plot(plot_x,
            plot_y[i].real,
            #xerr = ,
            #yerr = ,
            #capsize = ,
            #marker = 'o',
            #markersize = 2,
            color = colour,
            markerfacecolor = 'black',
            #linestyle = '-',
            #label = 'asdef'
            )

plt.xlim(0,RLC*10)
plt.xlim(plot_x[0],plot_x[-1])
plt.ylim(-1,4)

plt.show()
