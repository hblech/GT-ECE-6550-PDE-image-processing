# coding: utf-8

import numpy as np
import cv2
from skimage.measure import compare_ssim, compare_mse, compare_psnr
from utils.super_resolution_utils import *
from utils.cpp_grad_utils_compiled import *


# Open original image and region of interest
im_bgr = cv2.imread('hedgehog.jpg')
im = im_bgr.astype(np.float)

roi = im[500:900, 800:1300]
#roi = im

# Set variables to generate low resolution images
q = 2
nb_lr_im = 8
noise = 0.1*(np.max(roi) - np.min(roi))

# Set variables for super resolution
iterations = 200
l = 6
beta = 0.8
#dt = 1/(np.abs(l)*max(4/beta, 2))
dt=0.033
version_tau = 1
C = 1/6
exp_file = 'experiences.txt'


# Begin experiment report
with open(exp_file, 'a') as f:
    f.write("#Conditions de l'expérience:\n")
    f.write("#q={}, nb_lr_images={}, dt = {}, lambda={}, beta={}, nb_iterations={}, version_tau={}, C={}\n".format(q, nb_lr_im, dt, l, beta, iterations, version_tau, C))

# Create low resolution images and store them somewhere on disk
lr_images = createLRSamples(roi, nb_lr_images=nb_lr_im, q=q, noise_variance=noise)

for i in range(lr_images.shape[0]):
    cv2.imwrite('lr_images/lr_'+str(i)+'.png', lr_images[i])
cv2.imwrite('roi.png', roi)

# Create new (empty) high resolution image
hr_dims = get_hr_dims(lr_images.shape, q)
hr_image = np.zeros(hr_dims)

comparison_gradient = 0
denoising_gradient = 0
smoothing_gradient = 0
tau = 0

for iteration in range(iterations+1):

    # Periodically measure metrics and write current image to disk
    if iteration % 5 == 0:
        mse = compare_mse(roi, hr_image)
        ssim = compare_ssim(roi, hr_image, multichannel=True)
        psnr = compare_psnr(roi, hr_image, data_range=255.0)
        name = 'hr_images/hr_'+str(iteration)+'.png'
        cv2.imwrite(name, hr_image)
        with open(exp_file, 'a') as f:
            f.write("{}\t{}\t{}\t{}\t{}\n".format(iteration, mse, ssim, psnr, name))
            # Catches anormal cases
            if(mse > 1e8):
                f.write("Experience diverging, stopping now\n")
                raise ValueError("The algorithm is diverging")
            if(np.isnan(mse) or np.isnan(ssim) or np.isnan(psnr)):
                f.write("Computation error\n")
                raise ValueError("Computation error")


    # Compute directional order 1 and 2 gradients
    Ix = grad_x(hr_image)
    Iy = grad_y(hr_image)
    Ixy = grad_y(Ix)
    Ixx = grad_xx(hr_image)
    Iyy = grad_yy(hr_image)

    #Compute data fidelity energy gradient, denoising energy gradient, smoothing energy gradient
    comparison_gradient = data_fidelity_gradient_2(hr_image, lr_images, q)
    denoising_gradient = TV_regularization(Ix, Iy, Ixy, Ixx, Iyy, beta)
    smoothing_gradient = heat_gradient(Ix, Iy, Ixy, Ixx, Iyy)

    #Compute weighting factor
    if version_tau == 1:
        tau = compute_tau_1(Ix, Iy, C)
    elif version_tau == 2:
        tau = compute_tau_2(Ix, Iy)
    elif version_tau == 3:
        tau = compute_tau_3(Ix, Iy, C)

    #Update HR image: apply gradients to image
    hr_image -= dt*(comparison_gradient
                    + l*(tau*denoising_gradient
                    + (1-tau)*smoothing_gradient)
                   )