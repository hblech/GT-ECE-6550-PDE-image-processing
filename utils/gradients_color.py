# coding: utf-8

import numpy as np

#pythran export grad_x(float[][][])
def grad_x(img):
    """
    Compute gradient on x direction using central difference approximation
    """
    gradx = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y, size_z) = img.shape
    
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
    
    for z in range(size_z):
        for y in range(size_y):
            gradyy[0, y, z] = (2*img[0,y, z] - 5*img[1,y, z] + 4*img[2,y, z] - img[3,y, z])/2.0
            gradyy[-1, y, z] = (2*img[-1,y, z] - 5*img[-2,y, z] + 4*img[-3,y, z] - img[-4,y, z])/2.0
            for x in range(1,size_x-1):
                gradyy[x, y, z] = (img[x-1,y, z]  - 2*img[x,y, z] + img[x+1, y, z])
    return gradyy

#pythran export heat_equation(float[][][])
def heat_equation(img):
    It = np.zeros(img.shape)
    (size_x, size_y, size_z) = img.shape
    
    Ix = grad_x(img)
    Iy = grad_y(img)
    Ixy = grad_y(Ix)
    Ixx = grad_xx(img)
    Iyy = grad_yy(img)
    
    for z in range(size_z):
        for x in range(size_x):
            for y in range(size_y):
                if Ix[x,y,z] == 0 and Iy[x,y,z] == 0:
                    It[x,y,z] = - 2*Ix[x,y,z]*Iy[x,y,z]*Ixy[x,y,z]
                else:
                    It[x,y,z] = (Ix[x,y,z]*Ix[x,y,z]*Iyy[x,y,z] - 2*Ix[x,y,z]*Iy[x,y,z]*Ixy[x,y,z] + Iy[x,y,z]*Iy[x,y,z]*Ixx[x,y,z]) / (Ix[x,y,z]*Ix[x,y,z]+Iy[x,y,z]*Iy[x,y,z])
    return It

#pythran export TV_regularization(float[][][], float)
def TV_regularization(img, beta=0.1):
    It = np.zeros(img.shape)
    (size_x, size_y, size_z) = img.shape
    
    Ix = grad_x(img)
    Iy = grad_y(img)
    Ixy = grad_y(Ix)
    Ixx = grad_xx(img)
    Iyy = grad_yy(img)
    
    for z in range(size_z):
        for x in range(size_x):
            for y in range(size_y):
                It[x,y,z] = (Iyy[x,y,z] * Ix[x,y,z]**2 - 2*Ix[x,y,z]*Iy[x,y,z]*Ixy[x,y,z] + Ixx[x,y,z] * Iy[x,y,z]**2 + (Ixx[x,y,z]+Iyy[x,y,z]) * beta**2 ) / (Ix[x,y,z]**2+Iy[x,y,z]**2 + beta**2)**(3/2)
    return It