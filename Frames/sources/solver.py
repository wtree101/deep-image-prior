import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/home/twubi/deep-image-prior/Frames/sources/')
from tool import *
from multiprocessing import Pool, cpu_count
import time

def solver(g,Image,solver_setting,output_setting):
    """ 
    g: noisy image

    """
    K = solver_setting['iteration_num']
    iteration_count = 0
    num_filter = solver_setting['num_filter']
    r = solver_setting['kernel_size']
    Lambda = solver_setting['Lambda']
    r_2 = r**2
    D = g.shape[0]
    N = D**2

    verbose = output_setting['verbose']
    output_interval = output_setting['output_interval']
    output_folder = output_setting['path_output']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    snr_list = []
    err_list = []
    ## Frame: convolution kernel
    A = np.zeros((num_filter,r,r))  #2D r by r
    A_flat_mat = np.zeros((r_2,num_filter))
   #G = np.zeros((N,r,r))  #?  r*r patches of g, from conv with a 
    G_flat_mat = np.zeros((r_2,N))
    G_flat_mat_T= np.zeros((N,r_2))
    V_coded = np.zeros((num_filter,D,D))
    #V_coded_flat = np.zeros((N,num_filter))
    V_flat_mat = np.zeros((num_filter,N)) # V in paper
    Vc_flat_mat_T = np.zeros((num_filter,N))  #with T effect, V in paper
    #GTV = np.zeros((r_2,r_2)) #
    VGT = np.zeros((num_filter,r_2)) #  num_filter,r_2

    # Initialization
    print('Initialization! at' + str(time.ctime(time.time())))
    G_flat_mat,A = flat(g,A) #0-1bases A initialization and G_flat_mat
    G_flat_mat_T = G_flat_mat.T
    A_flat_mat = square_to_vector(A, A_flat_mat)

    print('Iteration! at' + str(time.ctime(time.time())) )
    for iter in range(K):
       #V_coded = W_analysis(A, g, V_coded)
        #V_coded = A_flat_mat.T @ G_flat_mat
        # V_coded = thresholding(Lambda, V_coded)
        # Vc_flat_mat_T = square_to_vector(V_coded, V_flat_mat).T
        # V_flat_mat = Vc_flat_mat_T
        V_wg = A_flat_mat.T @ G_flat_mat
        #
        err_list.append(loss(Lambda,V_flat_mat,V_wg))

        V_flat_mat = thresholding(Lambda, V_wg)

        err_list.append(loss(Lambda,V_flat_mat,V_wg))

        VGT = V_flat_mat @ G_flat_mat_T

        u, s, vh = np.linalg.svd(VGT, full_matrices=False, compute_uv=True, hermitian=False)
        A_flat_mat = (vh.T @ u.T) / r
        A = vector_to_square(A_flat_mat, A)




        if verbose == True:
            #V_coded = vector_to_square(V_flat_mat.T,V_coded)
           
            if iter % output_interval == 0:
                denoise_image = W_synthesis(A,thresholding(Lambda,W_analysis(A,g,V_coded)),g)
            #denoise_image = W_synthesis(A, thresholding(Lambda, V_coded), g)
                snr_list.append(snr(Image - denoise_image, Image))
                image_filename = f"denoised_image_{iteration_count}.png"  # 图片文件名，根据迭代次数进行命名
                image_filepath = os.path.join(output_folder, image_filename)
                plt.imshow(denoise_image, cmap='gray')
                plt.title(f'Denoised Image (Iteration: {iteration_count})')
                plt.axis('off')
                plt.savefig(image_filepath)
                plt.close()
                iteration_count += 1

        if iter % output_interval == 0:
            print(iter)
    
    if verbose == True:
        snr_plot_filename = "snr_plot.png"  # 图表文件名
        snr_plot_filepath = os.path.join(output_folder, snr_plot_filename)
        plt.plot(snr_list, label='Snr')
        plt.xlabel('Iteration')
        plt.ylabel('Snr')
        plt.title('Snr vs. Iteration')
        plt.legend()
        plt.savefig(snr_plot_filepath)
        plt.close()


    
    return A,err_list