# coding: utf-8

import numpy as np
import cv2
from skimage.measure import compare_ssim, compare_mse, compare_psnr
from utils.super_resolution_utils import *
from utils.cpp_grad_utils import *


# Open original image
im_bgr = cv2.imread('hedgehog.jpg')
im = im_bgr.astype(np.float)

roi = im[500:900, 800:1300]

# Set important variables
iterations = 400
q = 2
nb_lr_im = 8
noise = 0.1*(np.max(roi) - np.min(roi))
l = 5.0
beta = 0.8
dt = 1/(np.abs(l)*max(4/beta, 2))
version_tau = 1
C = 0.25
exp_file = 'experiences.txt'

with open(exp_file, 'a') as f:
    f.write("#Conditions de l'expérience:\n")
    f.write("#q={}, nb_lr_images={}, dt = {}, lambda={}, beta={}, nb_iterations={}, version_tau={}, C={}\n".format(q, nb_lr_im, dt, l, beta, iterations, version_tau, C))

# Create small images and store them somewhere on disk. Should work for gray and color images
lr_images = createLRSamples(roi, nb_lr_images=nb_lr_im, q=q, noise_variance=noise)
for i in range(lr_images.shape[0]):
    cv2.imwrite('lr_images/lr_'+str(i)+'.png', lr_images[i])
cv2.imwrite('roi.png', roi)
# Create new empty HR image. Should work for color and gray images
hr_dims = get_hr_dims(lr_images.shape, q)
hr_image = np.zeros(hr_dims)

comparison_gradient = 0
denoising_gradient = 0
smoothing_gradient = 0
tau = 0

for iteration in range(iterations+1):

    # Save result
    if iteration % 5 == 0:
        mse = compare_mse(roi, hr_image)
        ssim = compare_ssim(roi, hr_image, multichannel=True)
        psnr = compare_psnr(roi, hr_image, data_range=255.0)
        name = 'hr_images/hr_'+str(iteration)+'.png'
        #TODO: make matplotlib graph with metrics in title ?
        cv2.imwrite(name, hr_image)
        with open(exp_file, 'a') as f:
            f.write("{}\t{}\t{}\t{}\n".format(iteration, mse, ssim, psnr))
            f.write("\t\t{}\t{}\t{}\t{}\t{}\t{}\n".format(np.min(dt*comparison_gradient), np.max(dt*comparison_gradient), np.min(-dt*l*tau*denoising_gradient), np.max(-dt*l*tau*denoising_gradient), np.min(-dt*l*(1-tau)*smoothing_gradient), np.max(-dt*l*(1-tau)*smoothing_gradient)))
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
    hr_image -= dt*(comparison_gradient + l*(tau*denoising_gradient + (1-tau)*smoothing_gradient))

    #TODO: put this inside a compiled function. But data_fidelity_gradient is problematic
    # hr_image -= dt*compute_gradient(hr_image, comparison_gradient, version_tau, l, C, beta)
