# coding: utf-8

import numpy as np


#pythran export grad_x(float[][][])
def grad_x(img):
    """
    Compute gradient on x direction using central difference approximation
    """
    gradx = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y, size_z) = img.shape

    #omp parallel for
    for z in range(size_z):
        for x in range(size_x):
            gradx[x, 0, z] = (-3*img[x,0, z] - img[x,1, z] + 4*img[x,2, z])/2.0
            gradx[x, -1, z] = (3*img[x,-1, z] - 4*img[x,-2, z] + img[x,-3, z])/2.0
            for y in range(1,size_y-1):
                gradx[x, y, z] = ( -img[x,y-1, z] + img[x, y+1, z])/2.0
    return gradx

#pythran export grad_y(float[][][])
def grad_y(img):
    """
    #Compute gradient on x direction using central difference approximation
    """
    grady = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y, size_z) = img.shape

    #omp parallel for
    for z in range(size_z):
        for y in range(size_y):
            grady[0, y, z] = (-3*img[0,y, z] - img[1,y, z] + 4*img[2,y, z])/2.0
            grady[-1, y, z] = (3*img[-1,y, z] - 4*img[-2,y, z] + img[-3,y, z])/2.0
            for x in range(1,size_x-1):
                grady[x, y, z] = ( -img[x-1,y, z] + img[x+1, y, z])/2.0
    return grady

#pythran export grad_xx(float[][][])
def grad_xx(img):
    """
    #Compute 2nd gradient on x direction using central difference approximation
    """
    gradxx = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y, size_z) = img.shape

    #omp parallel for
    for z in range(size_z):
        for x in range(size_x):
            gradxx[x, 0, z] = (2*img[x,0, z] - 5*img[x,1, z] + 4*img[x,2, z] - img[x,3, z])/2.0
            gradxx[x, -1, z] = (2*img[x,-1, z] - 5*img[x,-2, z] + 4*img[x,-3, z] - img[x,-4, z])/2.0
            for y in range(1,size_y-1):
                gradxx[x, y, z] = (img[x,y-1, z]  - 2*img[x,y, z] + img[x, y+1, z])
    return gradxx

#pythran export grad_yy(float[][][])
def grad_yy(img):
    """
    #Compute 2nd gradient on x direction using central difference approximation
    """
    gradyy = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y, size_z) = img.shape

    #omp parallel for
    for z in range(size_z):
        for y in range(size_y):
            gradyy[0, y, z] = (2*img[0,y, z] - 5*img[1,y, z] + 4*img[2,y, z] - img[3,y, z])/2.0
            gradyy[-1, y, z] = (2*img[-1,y, z] - 5*img[-2,y, z] + 4*img[-3,y, z] - img[-4,y, z])/2.0
            for x in range(1,size_x-1):
                gradyy[x, y, z] = (img[x-1,y, z]  - 2*img[x,y, z] + img[x+1, y, z])
    return gradyy


#pythran export TV_regularization(float[][][], float[][][], float[][][], float[][][], float[][][], float)
def TV_regularization(Ix, Iy, Ixy, Ixx, Iyy, beta=0.1):
    gradE = np.zeros_like(Ix)
    (size_x, size_y, size_z) = Ix.shape

    #omp parallel for
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                gradE[x,y,z] = -(Iyy[x,y,z] * Ix[x,y,z]**2 - 2*Ix[x,y,z]*Iy[x,y,z]*Ixy[x,y,z] + Ixx[x,y,z] * Iy[x,y,z]**2 + (Ixx[x,y,z]+Iyy[x,y,z]) * beta**2 ) / (Ix[x,y,z]**2 + Iy[x,y,z]**2 + beta**2)**(3/2)
    return gradE

#pythran export heat_gradient(float[][][], float[][][], float[][][], float[][][], float[][][])
def heat_gradient(Ix, Iy, Ixy, Ixx, Iyy):
    gradE = np.zeros_like(Ix)
    (size_x, size_y, size_z) = Ix.shape

    #omp parallel for
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                if Ix[x,y,z] == 0 and Iy[x,y,z] == 0:
                    gradE[x,y,z] = 2*Ix[x,y,z]*Iy[x,y,z]*Ixy[x,y,z]
                else:
                    gradE[x,y,z] = -(Ix[x,y,z]**2*Iyy[x,y,z] - 2*Ix[x,y,z]*Iy[x,y,z]*Ixy[x,y,z] + Iy[x,y,z]**2*Ixx[x,y,z]) / (Ix[x,y,z]**2 + Iy[x,y,z]**2)
    return gradE

#pythran export compute_tau_1(float[][][], float[][][], float)
def compute_tau_1(Ix, Iy, C):
    gradI = np.sqrt(Ix**2 + Iy**2)
    m = np.max(gradI)
    tau = np.zeros_like(Ix)

    #omp parallel for
    for x in range(gradI.shape[0]):
        for y in range(gradI.shape[1]):
            for z in range(gradI.shape[2]):
                if gradI[x,y,z] < C*m:
                    tau[x,y,z] = np.sin(np.pi*gradI[x,y,z]/m/2/C)
                else:
                    tau[x,y,z] = 1
    return tau

#pythran export compute_tau_2(float[][][], float[][][])
def compute_tau_2(Ix, Iy):
    gradI = np.sqrt(Ix**2 + Iy**2)
    m = np.max(gradI)
    tau = np.zeros_like(gradI)
    
    #omp parallel for
    for x in range(gradI.shape[0]):
        for y in range(gradI.shape[1]):
            for z in range(gradI.shape[2]):
                tau[x,y,z] = (gradI[x,y,z]**2)/(m**2)
    return tau

#pythran export compute_tau_3(float[][][], float[][][], float)
def compute_tau_3(Ix, Iy, C):
    gradI = np.sqrt(Ix**2 + Iy**2)
    m = np.max(gradI)
    tau = np.zeros_like(Ix)

    #omp parallel for
    for x in range(gradI.shape[0]):
        for y in range(gradI.shape[1]):
            for z in range(gradI.shape[2]):
                if gradI[x,y,z] < C*m:
                    tau[x,y,z] = (1-np.cos(np.pi*gradI[x,y,z]/m/C))/2
                else:
                    tau[x,y,z] = 1
    return tau

#pythran export compute_gradient(float[][][], float[][][], int, float, float, float)
def compute_gradient(hr_image, comparison_gradient, version_tau, l, C, beta):

    #Compute gradients
    Ix = grad_x(hr_image)
    Iy = grad_y(hr_image)
    Ixy = grad_y(Ix)
    Ixx = grad_xx(hr_image)
    Iyy = grad_yy(hr_image)

    #Compute weighting factor
    if version_tau == 1:
        tau = compute_tau_1(Ix, Iy, C)
    elif version_tau == 2:
        tau = compute_tau_2(Ix, Iy)
    elif version_tau == 3:
        tau = compute_tau_3(Ix, Iy, C)

    #Compute denoising energy gradient and smoothing energy gradient
    denoising_gradient = TV_regularization(Ix, Iy, Ixy, Ixx, Iyy, beta)
    smoothing_gradient = heat_gradient(Ix, Iy, Ixy, Ixx, Iyy)

    gradient = np.zeros_like(comparison_gradient)
    #omp parallel for
    for x in range(comparison_gradient.shape[0]):
        for y in range(comparison_gradient.shape[1]):
            for z in range(comparison_gradient.shape[2]):
                gradient[x,y,z] = comparison_gradient[x,y,z] - l*(tau[x,y,z]*denoising_gradient[x,y,z] + (1-tau[x,y,z])*smoothing_gradient[x,y,z])

    return gradient

# pythran export upsample(float[][][], int, (int, int, int))
def upsample(lr_im, q, hr_shape):
    
    # Upsampling (D^T)
    temp_hr = np.zeros(hr_shape)

    #omp parallel for
    for i in range(lr_im.shape[0]):
        for j in range(lr_im.shape[1]):
            for k in range(lr_im.shape[2]):
                temp_hr[q*i, q*j, k] = lr_im[i,j,k]
    #temp_hr = 1*temp_hr # Unblurring (B^T)
    #temp_hr = 1*temp_hr # Unwarping (M^T)
    return temp_hr