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
#fname = '/home/twubi/deep-image-prior/Frames/images/im64.png'
image_name = 'barbara'
fname = os.path.join('/home/twubi/deep-image-prior/Frames/images/',image_name+'.png')
Image = mpimg.imread(fname=fname)
sigma = 20/255 #for im64
#sigma = 20/255
D = Image.shape[0] 
noisy_img = (Image + sigma*np.random.randn(D, D)); 	# add noise
noisy_img[noisy_img > 1] = 1; 
noisy_img[noisy_img < 0]   = 0; 

## Input 
#image g 
solver_setting = {
    'iteration_num': 40000,
    'num_filter': 16*16,  # Replace with the actual value
    'kernel_size': 16,  # Replace with the actual value
    'Lambda': 3.4 * sigma  # Replace with the actual value

}

output_setting = {
    'verbose': False,  # Replace with the actual value
    'output_interval': 100,  # Replace with the actual value
    'path_output': '/home/twubi/deep-image-prior/Frames/output3/'
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



g = noisy_img.copy()


A = solver(g,Image,solver_setting,output_setting)
np.save(os.path.join(filters_filepath, filters_filename), A)
print('Finish solving at' + str(time.ctime(time.time())))

V_coded = np.zeros((solver_setting['num_filter'],D,D))
denoise_image = W_synthesis(A,thresholding(2.7 * sigma,W_analysis(A,g,V_coded)),g)
print('Finish sythesis and denoising at'+str(time.ctime(time.time())))

plt.imshow(denoise_image, cmap='gray')
plt.title(f'Denoised Image (Snr: {snr(Image - denoise_image, Image)})')
plt.axis('off')
plt.savefig(os.path.join(image_filepath, image_filename))
plt.close()




