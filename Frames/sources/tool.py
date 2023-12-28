## W and thresh
import numpy as np
import scipy.signal as signal

def W_analysis(A,g,V):  # Sg(-.) correlation 0trans Wg
    num_filter = A.shape[0]
    for i in range(num_filter):
        V[i,:,:] = signal.correlate2d(g,A[i,:,:],'same','wrap')
    
    return V
        
        
def W_synthesis(A,V,g): #WTg
    num_filter = A.shape[0]
    D = g.shape[0]
    g_denoise = np.zeros((D,D))
    for i in range(num_filter):
        g_denoise += signal.convolve2d(V[i,:,:],A[i,:,:],'same','wrap')
    
    return g_denoise
    

def thresholding(Lambda,V):
    mask = (abs(V) < Lambda)
    V[mask] = 0
    return V


# see as numpy matrix
def square_to_vector(V_coded,V_flat_mat):
    # V_coded = np.zeros((num_filter,D,D))
    # V_flat_mat = np.zeros((num_filter,N))  #with T effect
    # if G image 
    num_filter = V_coded.shape[0]
    D = V_coded.shape[1]
    N = V_flat_mat.shape[1]
    return (V_coded.reshape(( num_filter,N))).T

def vector_to_square(V_flat_mat,V_coded): # V_codes change!
    # A = np.zeros((num_filter,r,r))  #2D r by r
    # A_flat_mat = np.zeros((r_2,num_filter))
    num_filter = V_coded.shape[0]
    r = V_coded.shape[1] #or D
    r_2 = r*r #or N = D*D
    # for i in range(num_filter):
    #     V_coded[i,:,:] = V_flat_mat[:,i].reshape((r,r))
    #V_coded = np.transpose(V_flat_mat.T.reshape((r,r,num_filter)),(2,0,1))
    V_coded = (V_flat_mat.T).reshape((num_filter,r,r))
    
    return V_coded
def flat(g,A):  # g -> G position patch ??  G r^2 * N
    #G = np.zeros((N,r,r))  #?  r*r patches of g, from conv with a 
    #only use the shape of A
    """
    Construct matrix from conv operator
    """
    D = g.shape[0]
    r = A.shape[1]
    r_2 = r*r
    N = D*D
    G2 = np.zeros((N,r,r))
    # Construc id matrix A2
    A2 = np.zeros((r,r,r,r)) 
    for i in range(r):
        for j in range(r):
            A2[i,j,i,j] = 1
    A2 = A2.reshape((r_2,r,r))
    G3 = np.zeros((r_2,D,D))
    G3 = W_analysis(A2,g,G3) #G3 r_2 D,D
    
    return G3.reshape(r_2,N),A2/r

def snr(signal, noise):
    signal = np.reshape(signal,-1)
    noise = np.reshape(noise,-1)
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = np.sum(noise ** 2) / len(noise)
    snr = -10 * np.log10(signal_power / noise_power)
    return snr

def loss(Lambda,V,V_wg):
    is_ones = (abs(V) > 1e-14)
    return (np.linalg.norm(V-V_wg)**2  + (Lambda**2) * is_ones.sum()) 
