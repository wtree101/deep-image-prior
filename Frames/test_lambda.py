import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import scipy.signal as signal
import math
import matplotlib.image as mpimg
import sys
sys.path.append('/home/twubi/deep-image-prior/Frames/sources/')
from solver import solver
import os
from tool import *
import time

image_name = 'barbara'
fname = os.path.join('/home/twubi/deep-image-prior/Frames/images/',image_name+'.png')
#fname = os.path.join('/home/twubi/deep-image-prior/Frames/images/',image_name+'.png')
Image = mpimg.imread(fname=fname)
sig = 20/255 #for im64

#sigma = 20/255
D = Image.shape[0] 
np.random.seed(42)
noisy_img = (Image + sig*np.random.randn(D, D)); 	# add noise
noisy_img[noisy_img > 1] = 1; 
noisy_img[noisy_img < 0]   = 0; 

plt.imshow(noisy_img, cmap='gray')
## Input 
#image g 
r = int(D/32)  #kersize /D/32   512/32 = 16
factor_list = [i for i in range(0, 14, 1)]
factor_list = [x  for x in factor_list]
sigma = sig / r     # sigma and kernel size
solver_setting = {
    'iteration_num': 15,
    'num_filter': r*r,  # Replace with the actual value
    'kernel_size': r,  # Replace with the actual value
    'Lambda': 0 #3.4 * sigma  # Replace with the actual value

}

output_setting = {
    'verbose': False,  # Replace with the actual value
    'output_interval': 5,  # Replace with the actual value
    'path_output': os.path.join('/home/twubi/deep-image-prior/Frames/Lambda_select/',image_name)
}

## Output
image_filename = f"denoised_image.png"  # 图片文件名，根据迭代次数进行命名
output_folder = output_setting['path_output']
image_filepath = os.path.join(output_folder, 'image')
filters_filename = f"filters.npy"
filters_filepath = os.path.join(output_folder, 'data')
for path in [output_folder,image_filepath,filters_filepath]:
    if not os.path.exists(path):
        os.makedirs(path)





for factor in factor_list:
    print(factor)
    image_filename = str(factor) + "denoised_image.png"  
    filters_filename = str(factor)+ "filters.npy"

    solver_setting['Lambda'] = sigma * factor
    g = noisy_img.copy()
    A,err_list = solver(g,Image,solver_setting,output_setting)
    np.save(os.path.join(filters_filepath, filters_filename), A)
    print('Finish solving at' + str(time.ctime(time.time())))

    V_coded = np.zeros((solver_setting['num_filter'],D,D))
    denoise_image = W_synthesis(A,thresholding(solver_setting['Lambda']/2,W_analysis(A,g,V_coded)),g)
    print('Finish sythesis and denoising at'+str(time.ctime(time.time())))

    image_filename = str(factor) + "denoised_image.png"
    plt.imshow(denoise_image, cmap='gray')
    plt.title(f'Denoised Image (Snr: {snr(Image - denoise_image, Image)})')
    plt.axis('off')
    plt.savefig(os.path.join(image_filepath, image_filename))
    plt.close()

    image_filename = str(factor) + "err.png"
    plt.plot(err_list)
    plt.title(f'loss')
    plt.axis('off')
    plt.savefig(os.path.join(image_filepath, image_filename))
    plt.close()









