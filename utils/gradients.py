# coding: utf-8

import numpy as np

#pythran export grad_x(float[][])
def grad_x(img):
    """
    Compute gradient on x direction using central difference approximation
    """
    gradx = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y) = img.shape
    
    for x in range(size_x):
        gradx[x, 0] = (-3*img[x,0] - img[x,1] + 4*img[x,2])/2.0
        gradx[x, -1] = (3*img[x,-1] - 4*img[x,-2] + img[x,-3])/2.0
        for y in range(1,size_y-1):
            gradx[x, y] = ( -img[x,y-1] + img[x, y+1])/2.0
        
    return gradx

#pythran export grad_y(float[][])
def grad_y(img):
    """
    #Compute gradient on x direction using central difference approximation
    """
    grady = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y) = img.shape
    
    for y in range(size_y):
        grady[0, y] = (-3*img[0,y] - img[1,y] + 4*img[2,y])/2.0
        grady[-1, y] = (3*img[-1,y] - 4*img[-2,y] + img[-3,y])/2.0
        for x in range(1,size_x-1):
            grady[x, y] = ( -img[x-1,y] + img[x+1, y])/2.0
        
    return grady

#pythran export grad_xx(float[][])
def grad_xx(img):
    """
    #Compute 2nd gradient on x direction using central difference approximation
    """
    gradxx = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y) = img.shape
    
    for x in range(size_x):
        gradxx[x, 0] = (2*img[x,0] - 5*img[x,1] + 4*img[x,2] - img[x,3])/2.0
        gradxx[x, -1] = (2*img[x,-1] - 5*img[x,-2] + 4*img[x,-3] - img[x,-4])/2.0
        for y in range(1,size_y-1):
            gradxx[x, y] = (img[x,y-1]  - 2*img[x,y] + img[x, y+1])
        
    return gradxx

#pythran export grad_yy(float[][])
def grad_yy(img):
    """
    #Compute 2nd gradient on x direction using central difference approximation
    """
    gradyy = np.zeros(img.shape, dtype=np.float)
    (size_x, size_y) = img.shape
    
    for y in range(size_y):
        gradyy[0, y] = (2*img[0,y] - 5*img[1,y] + 4*img[2,y] - img[3,y])/2.0
        gradyy[-1, y] = (2*img[-1,y] - 5*img[-2,y] + 4*img[-3,y] - img[-4,y])/2.0
        for x in range(1,size_x-1):
            gradyy[x, y] = (img[x-1,y]  - 2*img[x,y] + img[x+1, y])
        
    return gradyy

#pythran export heat_equation(float[][])
def heat_equation(img):
    It = np.zeros(img.shape)
    (size_x, size_y) = img.shape
    
    Ix = grad_x(img)
    Iy = grad_y(img)
    Ixy = grad_y(Ix)
    Ixx = grad_xx(img)
    Iyy = grad_yy(img)
    
    
    for x in range(size_x):
        for y in range(size_y):
            if Ix[x,y] == 0 and Iy[x,y] == 0:
                It[x,y] = - 2*Ix[x,y]*Iy[x,y]*Ixy[x,y]
            else:
                It[x,y] = (Ix[x,y]*Ix[x,y]*Iyy[x,y] - 2*Ix[x,y]*Iy[x,y]*Ixy[x,y] + Iy[x,y]*Iy[x,y]*Ixx[x,y]) / (Ix[x,y]*Ix[x,y]+Iy[x,y]*Iy[x,y])
    return It

#pythran export TV_regularization(float[][][], float)
def TV_regularization(img, beta=0.1):
    It = np.zeros(img.shape)
    (size_x, size_y) = img.shape
    
    Ix = grad_x(img)
    Iy = grad_y(img)
    Ixy = grad_y(Ix)
    Ixx = grad_xx(img)
    Iyy = grad_yy(img)
    
    for x in range(size_x):
        for y in range(size_y):
            It[x,y] = (Iyy[x,y] * Ix[x,y]**2 - 2*Ix[x,y]*Iy[x,y]*Ixy[x,y] + Ixx[x,y] * Iy[x,y]**2 + (Ixx[x,y]+Iyy[x,y]) * beta**2 ) / (Ix[x,y]**2+Iy[x,y]**2 + beta**2)**(3/2)
    return It