# TODO

# restrict surface plot to first brillouin zone

# fix graph, check vectors
# coupling between same spin different layer - layer 1 to 2 and reverse
# use same value in all

# This is the main file for self project

import numpy as np
import cmath
import math
import copy
import time
import tkinter
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, ImageMagickWriter
import collections


# --- Constants ---

# Simple conversions
def to_degrees(radians):
    return radians * 360 / (2 *np.pi)

def to_radians(degrees):
    return degrees * 2*np.pi/360

# Define the fitting parameters for the effective TBM Hamiltonian of monolayer NbSe2, energy parameter are in eV

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

# --- Vectors ---

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
    def __init__(self, vec: np.array):
        self.v = np.array(vec)
    
    def rotate(self, theta):
        self.v = np.matmul(rotation_matrix(theta), self.v)
        return self.v

    def rotate90(self, theta):
        self.v = np.matmul(rotation_matrix_90(theta), self.v)
        return self.v

    def len(self):
        return np.linalg.norm(self.v)

    def k(self, k: np.array):   # evaluate at some wavevector k, just an elementwise product
        return np.array([self.v[0] * k[0], self.v[1] * k[1]])

# for simplicity we will define the "default" primitive and reciprocal lattice vectors the same way they were defined before (before weekend of 13/11/21), self can be easily changed in the future if need be.

# Primitive lattice constant
PLC = 3.45e-10

# Primitive lattice vectors
PLV_1 = Vector([np.sqrt(3)*PLC/2, PLC/2])
PLV_2 = Vector([-np.sqrt(3)*PLC/2, PLC/2])

# --- Potential 'jumps' ---

# define alpha and beta from the wavevector k

def alpha(k: np.array):
    return 1/2*k[0]*PLC

def beta(k: np.array):
    return (math.sqrt(3)/2)*k[1]*PLC

"""
# defined from PLV
def alpha(k):
    return PLV_1.v[1]*k[0]

def beta(k):
    return PLV_1.v[0]*k[1]
"""

# define the 'hops' for each state (in order d_(z^2), d(x^2-y^2), d_(xy))
def V_0(k):
    tmp = epsilon_1
    tmp += 2*t_0 * (2*math.cos(alpha(k))*math.cos(beta(k)) + math.cos(2*alpha(k)))
    tmp += 2*r_0 * (2*math.cos(3*alpha(k))*math.cos(beta(k)) + math.cos(2*beta(k)))
    tmp += 2*u_0 * (2*math.cos(2*alpha(k))*math.cos(2*beta(k)) + math.cos(4*alpha(k)))
    return tmp

def Re_V_1(k):
    tmp = -2*math.sqrt(3)*t_2 * math.sin(alpha(k))*math.sin(beta(k))
    tmp += 2*(r_1+r_2) * math.sin(3*alpha(k))*math.sin(beta(k))
    tmp += -2*math.sqrt(3)*u_2 * math.sin(2*alpha(k))*math.sin(2*beta(k)) 
    return tmp

def Im_V_1(k):
    tmp = 2*t_1 * math.sin(alpha(k))*(2*math.cos(alpha(k)) + math.cos(beta(k)))
    tmp += 2*(r_1-r_2) * math.sin(3*alpha(k))*math.cos(beta(k))
    tmp += 2*u_1 * math.sin(2*alpha(k))*(2*math.cos(2*alpha(k)) + math.cos(2*beta(k)))
    return tmp

def V_1(k):
    return complex(Re_V_1(k), Im_V_1(k))

def Re_V_2(k):
    tmp = 2*t_2 * (math.cos(2*alpha(k)) - math.cos(alpha(k))*math.cos(beta(k)))
    tmp += (-2/math.sqrt(3))*(r_1 + r_2) * (math.cos(3*alpha(k))*math.cos(beta(k)) - math.cos(2*beta(k)))
    tmp += 2*u_2 * (math.cos(4*alpha(k)) - math.cos(2*alpha(k))*math.cos(2*beta(k)))
    return tmp

def Im_V_2(k):
    tmp = 2*math.sqrt(3)*t_1 * math.cos(alpha(k))*math.sin(beta(k))
    tmp += (2/math.sqrt(3))*(r_1-r_2) * math.sin(beta(k))*(math.cos(3*alpha(k)) + 2*math.cos(beta(k)))
    tmp += 2*math.sqrt(3)*u_1 * math.cos(2*alpha(k))*math.sin(2*beta(k))
    return tmp

def V_2(k):
    return complex(Re_V_2(k), Im_V_2(k))

def V_11(k):
    tmp = epsilon_2
    tmp += (t_11 + 3*t_22) * math.cos(alpha(k))*math.cos(beta(k))
    tmp += 2*t_11 * math.cos(2*alpha(k))
    tmp += 4*r_11 * math.cos(3*alpha(k))*math.cos(beta(k))
    tmp += 2*(r_11 + math.sqrt(3)*r_12) * math.cos(2*beta(k))
    tmp += (u_11 + 3*u_22) * math.cos(2*alpha(k))*math.cos(2*beta(k))
    tmp += 2*u_11 * math.cos(4*alpha(k))
    return tmp

def Re_V_12(k):
    tmp = math.sqrt(3)*(t_22-t_11) * math.sin(alpha(k))*math.sin(beta(k))
    tmp += 4*r_12 * math.sin(3*alpha(k))*math.sin(beta(k))
    tmp += math.sqrt(3)*(u_22-u_11) * math.sin(2*alpha(k))*math.sin(2*beta(k))
    return tmp

def Im_V_12(k):
    tmp = 4*t_12 * math.sin(alpha(k))*(math.cos(alpha(k)) - math.cos(beta(k)))
    tmp += 4*u_12 * math.sin(2*alpha(k))*(math.cos(2*alpha(k)) - math.cos(2*beta(k)))
    return tmp

def V_12(k):
    return complex(Re_V_12(k), Im_V_12(k))

def V_22(k) :
    tmp = epsilon_2
    tmp += ((3*t_11) + t_22) * math.cos(alpha(k))*math.cos(beta(k))
    tmp += 2*t_22 * math.cos(2*alpha(k))
    tmp += 2*r_11 * ((2*math.cos(3*alpha(k))*math.cos(beta(k))) + math.cos(2*beta(k)))
    tmp += (2/math.sqrt(3))*r_12 * ((4*math.cos(3*alpha(k))*math.cos(beta(k))) - math.cos(2*beta(k)))
    tmp += ((3*u_11) + u_22) * math.cos(2*alpha(k))*math.cos(2*beta(k))
    tmp += 2*u_22 * math.cos(4*alpha(k))
    return tmp

# --- Components of Hamiltonian ---

# Next we define the hamiltonian matrix of nearest neighbors TBM
def Hamiltonian_nearest_neighbors(k: np.array):
    return np.array([[V_0(k), V_1(k), V_2(k)],
                    [V_1(k).conjugate(), V_11(k), V_12(k)],
                    [V_2(k).conjugate(), V_12(k).conjugate(), V_22(k)]])

# L_z which describes the difference in energy due to spin orbit coupling
L_z = np.array([[0,0,0],
                [0,0,complex(0,-2)],
                [0,complex(0,2),0]])

# The pauli matrices sigma_0 and sigma_z
sigma_0 = np.array([[1,0],
                    [0,1]])

sigma_x = np.array([[0,1],
                    [1,0]])

sigma_z = np.array([[1,0],
                    [0,-1]])

top_left = np.array([[1,0],
                    [0,0]])

bottom_right = np.array([[0,0],
                        [0,1]])

# --- Hamiltonian ---

# the full hamiltonian
def Hamiltonian(k: np.array):
    return np.kron(sigma_0, Hamiltonian_nearest_neighbors(k)) + np.kron(sigma_z, (1/2)*lambda_SOC*L_z)

# rotation matrix for wavenumber vector to compensate for rotation of layers
# turns out this is equivalent to rotating the opposite direction with the normal rotation matrix
# so may remove this at some point later in favour of a simple -ve rotation
def k_rotation(k: np.array, phi):
    rot_mat = np.array([[math.cos(phi), math.sin(phi)],
                        [-math.sin(phi), math.cos(phi)]])
    return np.matmul(rot_mat, k)

# Class to manage the whole system - mostly for constructing the hamiltonian of the whole system (2 layers) and its eigenvalues
class Heterostructure:

    # heterostructure of lattices
    def __init__(self, plv1: Vector, plv2: Vector, plc, angle):
        
        # The lattices that make up the van der waals heterostructure
        self.lattices = np.array([])

        self.rotation = angle

        #primitive lattice vectors for 'universal k coordinate system'
        self.PLV_1 = plv1
        self.PLV_2 = plv2
    
        #primitive lattice constant
        self.PLC = plc
        self.RLC = (4*np.pi)/(np.sqrt(3)*self.PLC)

        four_pi_over_rt_three_RLC_sqrd = (4*np.pi)/(np.sqrt(3)*self.PLC*self.PLC)
        self.RLV_1 = Vector(four_pi_over_rt_three_RLC_sqrd*self.PLV_1.v)
        self.RLV_2 = Vector(-four_pi_over_rt_three_RLC_sqrd*self.PLV_2.v)

        # Initialise with uncoupled lattices
        self.is_coupled = False
        self.coupling = 0

    def gen_lattices(self):

        # create our two lattice layers, one rotated
        self.lattices = np.append(self.lattices, Lattice(self.PLC, self.PLV_1, self.PLV_2, 0, 0))
        self.lattices = np.append(self.lattices, Lattice(self.PLC, self.PLV_1, self.PLV_2, self.rotation, 1))

        # alternate rotation system - doesn't work currently
        #self.lattices = np.append(self.lattices, Lattice(self.PLC, self.PLV_1, self.PLV_2, -1/2*self.rotation, 0))
        #self.lattices = np.append(self.lattices, Lattice(self.PLC, self.PLV_1, self.PLV_2, 1/2*self.rotation, 1))

        # generates the reciprocal lattice & brilloin zone for each layer
        for lattice in self.lattices:
    
            # Generate each lattice by itself
            lattice.gen_reciprocal_lattice()
            lattice.gen_brilloin_zone_vectors()
            lattice.gen_brilloin_zone_path()

    def set_coupling(self, cpl):
        if cpl != 0:
            self.is_coupled = True
            self.coupling = cpl
        else:
            self.is_coupled = False
            self.coupling = 0

    # find Gm from RLV of each layer, there are 7
    def gen_coupling_vectors(self):
        
        G11 = self.lattices[0].RLV_1.v
        G12 = self.lattices[1].RLV_1.v
        G21 = self.lattices[0].RLV_2.v
        G22 = self.lattices[1].RLV_2.v

        GM0 = np.array([0,0])
        GM1 = np.array(G11 - G12)
        GM2 = np.array(G21 - G22)
        GM3 = np.array(GM2 - GM1)
        self.Gm = [GM0, GM1, GM2, GM3, -GM1, -GM2, -GM3]

    # Generate k' on other layer that a given k can couple to by k' = k + G - G' = k + Gm
    def gen_coupling_k(self, k: np.array):
        couple_k = np.array([np.empty(2)])
        for Gm in self.Gm:
            couple_k = np.concatenate((couple_k, np.array([k + Gm])))

        couple_k = np.delete(couple_k, 0, 0)
        return couple_k

    # Generates coupling hamiltonian between two k points on different layers, these are 12x12 matrices combined for the whole hamiltonian
    def gen_coupling_hamiltonian_diag(self, k: np.array):
        # Generates a 12x12 matrix using gen_layer_hamiltonian() for each lattice then combine using an inner product only if k1 = k2 = k (diagonal 12x12 blocks)
        hamiltonian = np.kron(top_left, self.lattices[0].gen_layer_hamiltonian(k)) + np.kron(bottom_right, self.lattices[1].gen_layer_hamiltonian(k))

        # Add interlayer coupling term
        if self.is_coupled:
            T = np.zeros((3,3))
            T[0][0] = self.coupling
            T = np.kron(sigma_0, T)
            T = np.kron(sigma_x, T)
            hamiltonian = hamiltonian + T

        return hamiltonian

    #off diagonal -> just coupling terms
    def gen_coupling_hamiltonian(self, k1: np.array, k2: np.array):
        hamiltonian = np.zeros((12,12))
        
        # Add interlayer coupling term
        if self.is_coupled:
            T = np.zeros((3,3))
            T[0][0] = self.coupling
            T = np.kron(sigma_0, T)
            T = np.kron(sigma_x, T)
            hamiltonian = hamiltonian + T

        return hamiltonian

    def gen_hamiltonian(self, k: np.array):
        kc = self.gen_coupling_k(k)
        gch = self.gen_coupling_hamiltonian
        gchd = self.gen_coupling_hamiltonian_diag
        zero = np.zeros((12,12))
        hamiltonian = np.block([
                    [gchd(kc[0]),       gch(kc[0],kc[1]),   gch(kc[0],kc[2]),   gch(kc[0],kc[3]),   gch(kc[0],kc[4]),   gch(kc[0],kc[5]),   gch(kc[0],kc[6])],
                    [gch(kc[1],kc[0]),  gchd(kc[1]),        gch(kc[1],kc[2]),   zero,               zero,               zero,               gch(kc[1],kc[6])],
                    [gch(kc[2],kc[0]),  gch(kc[2],kc[1]),   gchd(kc[2]),        gch(kc[2],kc[3]),   zero,               zero,               zero            ],
                    [gch(kc[3],kc[0]),  zero,               gch(kc[3],kc[2]),   gchd(kc[3]),        gch(kc[3],kc[4]),   zero,               zero            ],
                    [gch(kc[4],kc[0]),  zero,               zero,               gch(kc[4],kc[3]),   gchd(kc[4]),        gch(kc[4],kc[5]),   zero            ],
                    [gch(kc[5],kc[0]),  zero,               zero,               zero,               gch(kc[5],kc[4]),   gchd(kc[5]),        gch(kc[5],kc[6])],
                    [gch(kc[6],kc[0]),  gch(kc[6],kc[1]),   zero,               zero,               zero,               gch(kc[6],kc[5]),   gchd(kc[6])     ]
                    ])

        return(hamiltonian)
    
    #generate set of evectors for each of the 14 k states for projection make 84 long with zeros elsewhere
    def gen_state_evectors(self, k: np.array):
        AllEVecs = collections.deque([])
        kc = self.gen_coupling_k(k)
        for state in np.arange(0,7,1):
            for l in np.arange(0,2,1):
                h = self.lattices[l].gen_layer_hamiltonian(kc[state])
                EVal, EVecs = np.linalg.eigh(h)
                pos14Vec = state*2 + l      #position in 14*6 length vector, 0 indexed i.e. 0-13
                #determine number of zeros before and after to make 84 size vector
                preZeros = np.zeros((pos14Vec)*6)
                afterZeros = np.zeros((13 -pos14Vec)*6)
                
                for EVec in EVecs:
                    EVec = np.concatenate((preZeros, EVec), axis=None)
                    EVec = np.concatenate((EVec, afterZeros), axis=None)
                    AllEVecs.append(EVec)
                
        return np.asarray(AllEVecs)
        

    # Brilloin zone in our 'universal' k coordinate system
    def gen_brilloin_zone_vectors(self):

        self.Gamma_to_M = Vector((1.0/2.0) * self.RLV_1.v)

        self.Gamma_to_K = Vector((1.0/3.0) * (self.RLV_1.v + self.RLV_2.v))

        self.M_to_K = Vector((-1.0/6.0)*self.RLV_1.v + (1.0/3.0)*self.RLV_2.v)

        self.M_to_K_prime = Vector((1.0/6.0)*self.RLV_1.v - (1.0/3.0)*self.RLV_2.v)

        #to avoid rotating twice
        self.K_to_M = Vector(self.M_to_K_prime.v)

        self.Gamma_to_K_prime = Vector((2.0/3.0)*self.RLV_1.v - (1.0/3.0)*self.RLV_2.v)

        self.K_prime_to_Gamma = Vector((-2.0/3.0)*self.RLV_1.v + (1.0/3.0)*self.RLV_2.v)
        return 0

    # equilateral triangles path in brilloin zone
    def gen_brilloin_zone_path(self):
        self.brillouin_path = self.lattices[0].Gamma_to_Gamma + self.lattices[1].Gamma_to_Gamma
        #self.brillouin_path = self.lattices[0].M_to_M + self.lattices[1].M_to_M

        """
        # test these make a closed loop - returns 10-7 error in x, which is negligible
        tmp = np.array([0.0,0.0])
        print("\ntmp = ", tmp)
        for v in self.path:
            tmp += v.v
            print("\nvector = ", v.v)
            print("\ntmp = ", tmp)

        print("self should be equal to zero: ", tmp)
        """

    def gen_eigenvalues(self, brillouin_path):
        # init arrays for evalues of right shape
        Energy = []
        EVec = []
        SEVec = [] #evectors of individual states for some k
        for i in np.arange(0,84,1):
            Energy.append(collections.deque([]))

        #print(np.shape(Energy))

        Path_x = np.array([])
        Path_y = np.array([])

        # 'Path' is the x axis in the energy band graph
        Path = np.array([])

        Path_Offset = 0 #for plotting x axis

        #assuming starting from point M
        #k_last = self.Gamma_to_M.v

        k_last = np.array([0.0,0.0])

        self.plot_corners = np.array([])
        Path_Offset += np.linalg.norm(k_last)
        #print("\nstarting position = ", k_last/self.RLC)
        #print("\ninitial x axis 'distance' travelled = ", Path_Offset)

        for vectors in brillouin_path:
            #print("\nmoving along vector: ", vectors.v)
            for x in np.arange(0, 1, 0.01):

                k_step = x*vectors.v + k_last

                # for plotting the path in the brillouin zone
                Path_x = np.append(Path_x, k_step[0])
                Path_y = np.append(Path_y, k_step[1])

                Path = np.append(Path, x*vectors.len() + Path_Offset)
                eValues, eVectors = np.linalg.eigh(self.gen_hamiltonian(k_step))   #evalues and evectors
                #eValues.sort()

                for i in np.arange(0, np.size(Energy, axis = 0), 1):
                    Energy[i].append(eValues.real[i])
                
                EVec.append(eVectors)
                SEVec.append(self.gen_state_evectors(k_step))

            k_last += vectors.v

            Path_Offset += vectors.len()
            self.plot_corners = np.append(self.plot_corners, Path_Offset)
            #print("total x axis 'distance' travelled = ", Path_Offset)


        self.eValues = np.asarray(Energy)
        self.eVectors = EVec
        self.stateEVectors = SEVec

        self.projection = []
        for i in np.arange(0, np.shape(self.eVectors)[0]):
            self.projection.append(np.vdot(self.stateEVectors[i], self.eVectors[i]))

        print (np.shape(self.projection))


        self.path = Path
        self.path_x = Path_x
        self.path_y = Path_y

        print('Evalues shape: ', np.shape(self.eValues))
        print('Evectors shape: ', np.shape(self.eVectors))
        print('individual state Evectors shape: ', np.shape(self.stateEVectors))

        return Energy, Path

    #generates evalues for each layer individually then combines
    def gen_seperate_layer_evalues(self, brillouin_path):

        # this is not great but it doesn't work otherwise (numpy)
        Energy_1 = np.array([])
        Energy_2 = np.array([])
        Energy_3 = np.array([])
        Energy_4 = np.array([])
        Energy_5 = np.array([])
        Energy_6 = np.array([])
        Energy_7 = np.array([])
        Energy_8 = np.array([])
        Energy_9 = np.array([])
        Energy_10 = np.array([])
        Energy_11 = np.array([])
        Energy_12 = np.array([])
        Energy = [Energy_1,Energy_2,Energy_3,Energy_4,Energy_5,Energy_6, Energy_7, Energy_8, Energy_9, Energy_10, Energy_11, Energy_12]

        for lat in self.lattices:
            lat.gen_eigenvalues(brillouin_path)
            for i in np.arange(0,6,1):
                Energy[i + 6*lat.layerindex] = np.append(Energy[i + 6*lat.layerindex], lat.eValues[i])
    
        l1 = self.lattices[0]
        self.eValues = Energy
        self.path = l1.path
        self.path_x = l1.path_x
        self.path_y = l1.path_y
        self.plot_corners = l1.plot_corners
        print(np.size(self.eValues, axis = 0))
        print(np.size(self.eValues, axis = 1))

    # plot the path in k that is taken
    def plot_brillouin_zone_path(self):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

        ax.plot(self.path_x, 
                self.path_y,
                marker = 'o',
                color = 'black',
                markersize = 2)

        ax.plot(0,0, marker = 'o', color = 'red')

        for lattice in self.lattices:
            ax.plot(lattice.Gamma_to_M.v[0],lattice.Gamma_to_M.v[1], marker = 'o', color = 'red')
            ax.plot(lattice.Gamma_to_K.v[0],lattice.Gamma_to_K.v[1], marker = 'o', color = 'red')
            ax.plot(lattice.Gamma_to_K_prime.v[0],lattice.Gamma_to_K_prime.v[1], marker = 'o', color = 'red')

        plt.axhline(y=0, xmin=-2, xmax=2, color = 'green')
        plt.axvline(x=0, ymin=-2, ymax=2, color = 'green')

        plt.xlim(-self.RLC,self.RLC)
        plt.ylim(-self.RLC,self.RLC)

        plt.show()

    # plot the eigenvalues (electronic bands)
    def plot_eigenvalues(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

        # plot position of corners in triangular path taken
        for vline in self.plot_corners:
            plt.axvline(x=vline, ymin=-1, ymax=1, color = 'green')

        # plot fermi level
        plt.axhline(xmin = self.path[0], xmax = self.path[-1], y = 0, color = 'red')

        # only plot first 4 evalues (fermi level)
        for i in np.arange(0, np.size(self.eValues, axis = 0),1):
        #for i in np.arange(0,4,1):
            '''
            if (i % 2) == 0:
                colour = 'red'
            else:
                colour = 'blue'
            '''
            colour = 'blue'

            ax.plot(self.path,
                    self.eValues[i].real,
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
        
        plt.xlim(0,self.RLC*10)
        plt.xlim(self.path[0],self.path[-1])
        #plt.ylim(-1,1)
        #plt.ylim(-0.5, 1)

        ax.set_xlabel('Distance along path in $k$ space [m$^{-1}$]')
        ax.set_ylabel('Energy [eV]')
        ax.set_title(f"Energy band structure of bilayer NbSe$_2$ with twist {round(to_degrees(self.rotation))}$^\circ$")

        plt.show()

    """ Not relevant right now
    def gen_surface_evalues(self, lattice_index):
        # x,y is a plane described by the RLVs
        # find evalues for each k = (x,y), need to sort for lowest? evalues
        # return 3 vector? (x,y,E)

        Energy = np.array([])
        kx = np.array([])
        ky = np.array([])
        
        k = Vector([0,0])

        for rlv_1 in np.arange(0,1,0.02):
            for rlv_2 in np.arange (0,1,0.02):

                k = (rlv_1 * self.RLV_1.v) + (rlv_2 * self.RLV_2.v)

                hamiltonian = self.lattices[lattice_index].gen_layer_hamiltonian(k)
                eValues = np.linalg.eigvalsh(hamiltonian)
                eValues.sort()

                Energy = np.append(Energy, eValues[0].real)
                kx = np.append(kx, k[0])
                ky = np.append(ky, k[1])
        
        return [kx, ky, Energy]
    """


    def surface_z(self, xx, yy, lattice_index, evalue_num):
        z = np.empty((np.shape(xx)[0],np.shape(xx)[1]))

        for i in range(np.shape(xx)[1]):
            for j in range(np.shape(xx)[0]):
                k = np.array([xx[j,i], yy[j,i]])
                #hamiltonian = self.lattices[lattice_index].gen_layer_hamiltonian(k)
                hamiltonian = self.gen_hamiltonian(k)
                eValues = np.linalg.eigvalsh(hamiltonian)
                eValues.sort()
                z[j,i] = eValues[evalue_num]
        return z

    def plot_surface(self, plot_type, evalue_num):

        # define space to plot surface in
        #kx = np.linspace(0, self.RLC, 100)
        #ky = np.linspace(0, self.RLC, 100)

        # bounded by side length of triangle path
        kx = np.linspace(0, self.Gamma_to_K.len(), 100)
        ky = np.linspace(0, self.Gamma_to_K.len(), 100)

        # full brilloin zone
        #kx = np.linspace(-self.RLC, self.RLC, 100)
        #ky = np.linspace(-self.RLC, self.RLC, 100)

        # very wide range
        #kx = np.linspace(-2*self.RLC, 2*self.RLC, 100)
        #ky = np.linspace(-2*self.RLC, 2*self.RLC, 100)

        # create a mesh grid in kx, ky
        xx, yy = np.meshgrid(kx, ky)

        # returns z as a function of the grid
        z_1 = self.surface_z(xx, yy, 0, evalue_num)
        z_2 = self.surface_z(xx, yy, 1, evalue_num)

        fig = plt.figure(figsize = (8,8))
        ax = plt.axes(projection = '3d')

        if plot_type == 1 or plot_type == 3:
            ax.plot_surface(xx, yy, z_1, cmap = 'viridis', zorder = 1)
            #ax.plot_surface(xx, yy, z_2, cmap = 'magma', zorder = 1)

        # Plot the evalues just along the path like original plots
        if plot_type == 2:
            ax.plot(self.path_x, self.path_y, self.eValues[evalue_num], color = 'red', markersize = 5, zorder = 2)
        elif plot_type == 3:
            ax.plot(self.path_x, self.path_y, self.eValues[evalue_num]+0.5, color = 'red', markersize = 5, zorder = 2)


        ax.set_xlabel('$k_x$ [m$^{-1}$]')
        ax.set_ylabel('$k_y$ [m$^{-1}$]')
        ax.set_zlabel('Energy [eV]')
        ax.set_title('Lowest energy electronic band surface in brilloin zone')
        plt.show()

    def animate(self, intrvl):
        
        fig, ax = plt.subplots()

        # Array to store line2D objects
        evalue_plots = [0,0,0,0]
        
        # Store rotation in degrees before manipulation
        rtn = round(to_degrees(self.rotation))

        def anim(i):
            self.rotation = float(to_radians(i))
            self.lattices = np.array([])
            self.gen_lattices()
            self.gen_brilloin_zone_vectors()
            self.gen_brilloin_zone_path()
            plt_y, plt_x = self.gen_eigenvalues(self.brillouin_path)

            # On first run initialise plots
            if i == 0:
                for j in np.arange(0,4,1):
                    tmp, = ax.plot(plt_x,
                                    plt_y[j].real,
                                    color = 'blue',
                                    markerfacecolor = 'black',
                                    )

                    evalue_plots[j] = tmp
            else:
                for j in np.arange(0,4,1):
                    evalue_plots[j].set_ydata(plt_y[j].real)

            return evalue_plots

        ani = FuncAnimation(fig, anim, np.arange(0, rtn+1, intrvl), blit = True, interval = 50)

        # plot position of corners in triangular path taken
        for vline in self.plot_corners:
            plt.axvline(x=vline, ymin=-1, ymax=1, color = 'green')

        # plot fermi level
        plt.axhline(xmin = self.path[0], xmax = self.path[-1], y = 0, color = 'red')

        plt.xlim(0,self.RLC*10)
        plt.xlim(self.path[0],self.path[-1])
        plt.ylim(-1,1)

        ax.set_xlabel('Distance along path in $k$ space [m$^{-1}$]')
        ax.set_ylabel('Energy [eV]')
        ax.set_title(f"Energy band structure of bilayer NbSe$_2$ with twist {round(to_degrees(self.rotation))}$^\circ$")

        #plt.show()

        # Save as GIF
        writer = ImageMagickWriter(fps=10)
        #writer = PillowWriter(fps=10)
        ani.save("Animation.gif", writer = writer)

        plt.show()


# --- Lattice ---

# Class that describes a lattice, can generate reciprocal lattice & hamiltonian.
class Lattice:

    def __init__(self, plc, plv_1: Vector, plv_2: Vector, angle, layer_index):
        #print("\n= Generating lattice... =")

        #important for heterostructure
        self.rotation = angle
        self.layerindex = layer_index

        #primitive lattice constant
        self.PLC = plc

        #primitive lattice vectors
        self.PLV_1 = copy.copy(plv_1)
        self.PLV_2 = copy.copy(plv_2)

        #rotates lattice vectors accordingly
        if angle != 0:
            self.PLV_1.rotate(self.rotation)
            self.PLV_2.rotate(self.rotation)

    def gen_reciprocal_lattice(self):
        #print("\n= Generating reciprocal lattice... =")

        # reciprocal lattice constant
        self.RLC = (4*np.pi)/(np.sqrt(3)*self.PLC)

        # reciprocal lattice vectors
        # conditions to generate RLVs: ai dot bj = 2Pi delta(i,j)
        four_pi_over_rt_three_RLC_sqrd = (4*np.pi)/(np.sqrt(3)*self.PLC*self.PLC)
        self.RLV_1 = Vector(four_pi_over_rt_three_RLC_sqrd*self.PLV_1.v)
        self.RLV_2 = Vector(-four_pi_over_rt_three_RLC_sqrd*self.PLV_2.v)

    # Define the vectors that describe the paths for points Gamma, K, K', M in the brillouin zone
    def gen_brilloin_zone_vectors(self):

        self.Gamma_to_M = Vector((1.0/2.0)*self.RLV_1.v)

        self.Gamma_to_K = Vector((1.0/3.0) * (self.RLV_1.v + self.RLV_2.v))

        self.M_to_K = Vector((-1.0/6.0)*self.RLV_1.v + (1.0/3.0)*self.RLV_2.v)

        self.M_to_K_prime = Vector((1.0/6.0)*self.RLV_1.v - (1.0/3.0)*self.RLV_2.v)

        #to avoid rotating twice
        self.K_to_M = Vector(-self.M_to_K.v)

        self.K_to_K_prime = Vector(2*self.K_to_M.v)

        self.Gamma_to_K_prime = Vector((2.0/3.0)*self.RLV_1.v - (1.0/3.0)*self.RLV_2.v)

        self.K_prime_to_Gamma = Vector((-2.0/3.0)*self.RLV_1.v + (1.0/3.0)*self.RLV_2.v)

    def gen_brilloin_zone_path(self):
        self.Gamma_to_Gamma = [self.Gamma_to_K,
                                self.K_to_M,
                                self.K_to_M,
                                self.K_prime_to_Gamma,]

        self.M_to_M = [self.M_to_K_prime,
                        self.K_prime_to_Gamma,
                        self.Gamma_to_K,
                        self.K_to_M]

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

    # generate a nearest neighbors hamiltonian, must rotate k input to be of perspective of the layer (k_prime)
    # this is equaivalent to rotating the coordinate system (kx, ky) -> (k'x, k'y)
    def gen_layer_hamiltonian(self, k: np.array):
        k_prime = k_rotation(k, self.rotation)
        return Hamiltonian(k_prime)

    #get eigenvalues for one layer
    def gen_eigenvalues(self, brillouin_path):
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

        k_last = np.array([0.0,0.0])

        #assuming starting from point M - should rotate with the rest
        #k_last = self.Gamma_to_M.v


        self.plot_corners = np.array([])
        Path_Offset += np.linalg.norm(k_last)
        #print("\nstarting position (M) = ", k_last/self.RLC)
        #print("\ninitial x axis 'distance' travelled = ", Path_Offset)

        for vectors in brillouin_path:
            #print("\nmoving along vector: ", vectors.v)
            for x in np.arange(0, 1, 0.01):

                k_step = x*vectors.v + k_last

                # for plotting the path in the brillouin zone
                Path_x = np.append(Path_x, k_step[0]/self.RLC)
                Path_y = np.append(Path_y, k_step[1]/self.RLC)

                Path = np.append(Path, x*vectors.len() + Path_Offset)
                eValues = np.linalg.eigvalsh(self.gen_layer_hamiltonian(k_step))   #just evalues
                eValues.sort()

            
                for i in np.arange(0, 6, 1):
                    Energy[i] = np.append(Energy[i], eValues[i].real)

            k_last += vectors.v
            #print("\ncurrent point in k = ", k_last/self.RLC)

            Path_Offset += vectors.len()
            self.plot_corners = np.append(self.plot_corners, Path_Offset)
            #print("total x axis 'distance' travelled = ", Path_Offset)

        self.eValues = Energy
        self.eVectors = EigenVectors
        self.path = Path
        self.path_x = Path_x
        self.path_y = Path_y

        return Energy, Path, EigenVectors, Path_x, Path_y
        
    # plot the path in k that is taken
    def plot_brillouin_zone_path(self):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

        ax.plot(self.path_x, 
                self.path_y,
                marker = 'o',
                color = 'black',
                markersize = 2)

        ax.plot(0,0, marker = 'o', color = 'red')
        ax.plot(self.Gamma_to_M.v[0]/self.RLC,self.Gamma_to_M.v[1]/self.RLC, marker = 'o', color = 'red')
        ax.plot(self.Gamma_to_K.v[0]/self.RLC,self.Gamma_to_K.v[1]/self.RLC, marker = 'o', color = 'red')
        ax.plot(self.Gamma_to_K_prime.v[0]/self.RLC,self.Gamma_to_K_prime.v[1]/self.RLC, marker = 'o', color = 'red')

        plt.axhline(y=0, xmin=-2, xmax=2, color = 'green')
        plt.axvline(x=0, ymin=-2, ymax=2, color = 'green')

        plt.xlim(-2,2)
        plt.ylim(-2,2)

        plt.show()

    # plot the eigenvalues (electronic bands)
    def plot_eigenvalues(self):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

        # plot position of corners in triangular path taken
        for vline in self.plot_corners:
            plt.axvline(x=vline, ymin=-1, ymax=1, color = 'green')

        # plot fermi level
        plt.axhline(xmin = self.path[0], xmax = self.path[-1], y = 0, color = 'red')

        for i in np.arange(0,6,1):
            """
            if (i % 2) == 0:
                colour = 'red'
            else:
                colour = 'blue'
            """
            colour = 'blue'

            ax.plot(self.path,
                    self.eValues[i].real,
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

        plt.xlim(0,self.RLC*10)
        plt.xlim(self.path[0],self.path[-1])
        plt.ylim(-1,4)
        #plt.ylim(-1,1)

        ax.set_xlabel('Distance along path in $k$ space [m$^{-1}$]')
        ax.set_ylabel('Energy [eV]')
        ax.set_title('Energy band structure of NbSe$_2$')

        plt.show()

#using the heterostructres class
#myTwistedBilayer = Heterostructure(PLV_1,PLV_2,PLC,to_radians(30))
myTwistedBilayer = Heterostructure(PLV_1,PLV_2,PLC,to_radians(0))
myTwistedBilayer.gen_lattices()
#myTwistedBilayer.set_coupling(0.3)
myTwistedBilayer.set_coupling(0)
myTwistedBilayer.gen_brilloin_zone_vectors()
myTwistedBilayer.gen_brilloin_zone_path()
myTwistedBilayer.gen_coupling_vectors()
myTwistedBilayer.gen_eigenvalues(myTwistedBilayer.brillouin_path)
#myTwistedBilayer.plot_brillouin_zone_path()
#myTwistedBilayer.gen_seperate_layer_evalues(myTwistedBilayer.brillouin_path)
myTwistedBilayer.plot_eigenvalues()
#myTwistedBilayer.animate(5)
#myTwistedBilayer.plot_surface(1, 3)
#myTwistedBilayer.plot_surface(2, 3)

l1 = myTwistedBilayer.lattices[0]
l2 = myTwistedBilayer.lattices[1]

#l1.gen_eigenvalues(l1.M_to_M)
#l1.gen_eigenvalues(myTwistedBilayer.brillouin_path)
#l1.plot_eigenvalues()

#l2.gen_eigenvalues(l1.M_to_M)
#l2.gen_eigenvalues(myTwistedBilayer.brillouin_path)
#l2.plot_eigenvalues()

#l2.gen_eigenvalues(l1.Gamma_to_Gamma)
#l2.plot_eigenvalues()





