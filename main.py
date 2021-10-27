import numpy as np
import scipy
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
    return epsilon_1 + 2*t_0*(2*math.cos(alpha(k))*math.cos(beta(k))+math.cos(2*alpha(k))) + 2*r_0*(2*math.cos(3*alpha(k))*math.cos(beta(k))+math.cos(2*beta(k))) + 2*u_0*(2*math.cos(2*alpha(k))*math.cos(2*beta(k))+math.cos(4*alpha(k)))

def Re_V_1(k):
    return -2*math.sqrt(3)*t_2*math.sin(alpha(k))*math.sin(beta(k)) + 2*(r_1+r_2)*math.sin(3*alpha(k))*math.sin(beta(k)) - 2*math.sqrt(3)*u_2*math.sin(2*alpha(k))*math.sin(2*beta(k)) 

def Im_V_1(k):
    return 2*t_1*math.sin(alpha(k))*(2*math.cos(alpha(k))+math.cos(beta(k))) + 2*(r_1-r_2)*math.sin(3*alpha(k))*math.cos(beta(k)) + 2*u_1*math.sin(2*alpha(k))*(2*math.cos(2*alpha(k))+math.cos(2*beta(k)))

def V_1(k):
    return complex(Re_V_1(k), Im_V_1(k))

def Re_V_2(k):
    return 2 * t_2 * (math.cos(2 * alpha(k)) + math.cos(beta(k))) - (2 / math.sqrt(3)) * (r_1 + r_2) * ((math.cos(3 * alpha(k)) * math.cos(beta(k))) - math.cos(2 * beta(k))) + 2 * u_2 * (math.cos(4 * alpha(k)) - (math.cos(2 * alpha(k))*math.cos(2*beta(k))))

def Im_V_2(k):
    return 2 * math.sqrt(3) * t_1 * math.cos(alpha(k)) * math.sin(beta(k)) + (2 / math.sqrt(3)) * (r_1 - r_2) * math.sin(beta(k)) * (math.cos(3 * alpha(k)) + 2 * math.cos(beta(k))) + 2 * math.sqrt(3)* u_1 * math.cos(2 * alpha(k)) * math.sin(2 * beta(k))

def V_2(k):
    return complex(Re_V_2(k), Im_V_2(k))

def V_11(k):
    return epsilon_2 + (t_11 + 3*t_22)*math.cos(alpha(k))*math.cos(beta(k)) + 2*t_11*math.cos(2*alpha(k)) + 4*r_11*math.cos(3*alpha(k))*math.cos(beta(k)) + 2*(r_11+math.sqrt(3)*r_12)*math.cos(2*beta(k)) + (u_11+3*u_22)*math.cos(2*alpha(k))*math.cos(2*beta(k)) + 2*u_11*math.cos(4*alpha(k))

def Re_V_12(k):
    return math.sqrt(3)*(t_22-t_11)*math.sin(alpha(k))*math.sin(beta(k)) + 4*r_12*math.sin(3*alpha(k))*math.sin(beta(k)) + math.sqrt(3)*(u_22-u_11)*math.sin(2*alpha(k))*math.sin(2*beta(k))

def Im_V_12(k):
    return 4*t_12*math.sin(alpha(k))*(math.cos(alpha(k))-math.cos(beta(k))) + 4*u_12*math.sin(2*alpha(k))*(math.cos(2*alpha(k))-math.cos(2*beta(k)))

def V_12(k):
    return complex(Re_V_12(k), Im_V_12(k))

def V_22(k) :
    return epsilon_2 + ((3 * t_11) + t_22) * math.cos(alpha(k)) * math.cos(beta(k)) + (2 * t_22 * math.cos(2 * alpha(k))) + 2 * r_11 * ((2 * math.cos(3 * alpha(k)) * math.cos(beta(k))) + math.cos(2 * beta(k))) + (2 / math.sqrt(3)) * r_12 * ((4 * math.cos(3 * alpha(k)) * math.cos(beta(k))) - math.cos(2 * beta(k)))+ ((3 * u_11) + u_22) * math.cos(2 * alpha(k)) * math.cos(2 * beta(k)) + (2 * u_22 * math.cos(4 * alpha(k)))

# --- Components of Hamiltonian ---

# Next we define H_TNN the matrix of nearest neighbors

def Hamiltonian_nearest_neighbors(k):
    return np.array([[V_0(k), V_1(k), V_2(k)],
                    [V_1(k).conjugate(), V_11(k), V_12(k)],
                    [V_2(k).conjugate(), V_12(k).conjugate(), V_22(k)]])

#print(type(V_1(k)))

# L_z which describes the difference in energy due to spin orbit coupling

L_z = np.array([[0,0,0],
                [0,0,-2j],
                [0,2j,0]])

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
# gamma -> m = 1/2 b (b=reciprocal lattice constant)
# gamma -> k = b/sqrt(3)
# k -> m = b/2*sqrt(3)

# reciproal lattice constant
RLC = (4*np.pi)/(np.sqrt(3)*PLC)

# Primitive lattice vectors
def PLV_1(k: np.array):
    return np.array([np.sqrt(3)*PLC*k[0]/2, PLC*k[1]/2])

def PLV_2(k: np.array):
    return np.array([-np.sqrt(3)*PLC*k[0]/2, PLC*k[1]/2])

# Reciprocal lattice vectors
def RLV_1(k: np.array):
    return np.array([2*np.pi*k[0]/(np.sqrt(3)*PLC), 2*np.pi*k[1]/PLC])

def RLV_2(k: np.array):
    return np.array([-2*np.pi*k[0]/(np.sqrt(3)*PLC), 2*np.pi*k[1]/PLC])

print("PLC = ", PLC, "\nRLC = ", RLC)
print("|PLV| = ", np.linalg.norm(PLV_1([1,1])), "\n|RLV| = ", np.linalg.norm(RLV_1([1,1])))

# Next we define the vectors that describe the paths for the brillouin zone

def Gamma_to_M(k: np.array):
    return (1/2)*RLV_1(k)

def Gamma_to_K_prime(k: np.array):
    return (1/3)*(RLV_1(k)+RLV_2(k))

def M_to_K_prime(k: np.array):
    return -Gamma_to_M(k)+Gamma_to_K_prime(k)

def M_to_K(k: np.array):
    return -M_to_K_prime(k)

def Gamma_to_K(k: np.array):
    return Gamma_to_M(k)+M_to_K(k)

def K_to_Gamma(k: np.array):
    return -Gamma_to_K(k)

# So for the complete path M -> K -> Gamma -> K' -> M should be the 0 vector

#def full_path_vector(k):
#    return -M_to_K_prime(k) - Gamma_to_K(k) + Gamma_to_K_prime(k) - M_to_K_prime(k)

#M_to_M = [M_to_K, -Gamma_to_K, Gamma_to_K_prime, -M_to_K_prime]
M_to_M = [M_to_K, K_to_Gamma, Gamma_to_K_prime, M_to_K]

print("vector scale = ", np.linalg.norm(M_to_K([1,1])))

#M_to_M = [M_to_K]

#print("\nThe complete path M -> K -> Gamma -> K' -> M should be the 0 vector: ", Full_Path([1,1]))

# conditions ai dot bj = 2Pi delta(i,j)
# pairwise parallel or perpendicular

# define the reciprocal lattice constant

# define better transform from real to reciprocal

# find coordinates of M, K, K'


def Figure_d():
    Energy_1 = np.array([])
    Energy_2 = np.array([])
    Energy_3 = np.array([])
    Energy_4 = np.array([])
    Energy_5 = np.array([])
    Energy_6 = np.array([])
    Energy = [Energy_1,Energy_2,Energy_3,Energy_4,Energy_5,Energy_6]

    #These don't work
    #Energy = np.array([[],[],[],[],[],[]])
    #Energy = np.empty(shape = (6,), dtype = float)
    #print(np.shape(Energy))

    Path = np.array([])
    EigenVectors = np.array([[],[],[],[],[],[]], dtype=complex)

    k=[1,1]
    Path_Offset = 0
    for vectors in M_to_M:
        print("\ndoing path")
        for x in np.arange(0, np.linalg.norm(vectors([1,1])), RLC/20):
            #print("x = ", x)

            k_step = x*vectors([1,1])
            #eValues, eVectors = np.linalg.eig(Hamiltonian(k_step))

            Path = np.append(Path, x + Path_Offset)
            eValues, eVectors = np.linalg.eig(Hamiltonian(k_step))

            #print("energy = ", eValues.real)
            #print("evector = ", eVectors[0])

            #Energy[0] = np.append(Energy[0], eValues[0].real)
            #EigenVectors[0] = np.append(EigenVectors[0], eVectors[0])

            #eValues = [1,2,3,4,5,6]

            #Energy = np.append(Energy, [eValues])
            #print(np.shape(Energy))

            for i in np.arange(0, 6, 1):
                Energy[i] = np.append(Energy[i], eValues[i].real)
                #Energy[i] = np.append(Energy[i], [5.0])
                #Energy[i] = np.append(EigenVectors[i], eVectors[i])
        Path_Offset += np.linalg.norm(vectors([1,1]))
        print("new offset = ", Path_Offset)

    #print (Energy[0][0])
    #print (Energy)
    print (Path)
    #print (EigenVectors)

    return Energy, Path, EigenVectors

plot_y, plot_x, eigvec = Figure_d()

#print (plot_x, plot_y)

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)

for i in np.arange(0,6,1):
    if i < 3:
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

#plt.xlim(0,RLC*10)
plt.xlim(0,plot_x[-1])
plt.ylim(-1,4)

plt.show()
