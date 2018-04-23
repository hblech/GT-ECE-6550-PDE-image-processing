import numpy as np
from .cpp_grad_utils import upsample

def get_lr_dims(shape, q, nb_lr_images):
    hr_shape = list(shape)
    lr_shape = hr_shape
    for i in range(2):
        lr_shape[i] = int(lr_shape[i]/q)
    lr_shape.insert(0, nb_lr_images)
    return tuple(lr_shape)

def createLRSamples(image, q=2, nb_lr_images=8, noise_variance=10):
    """
    Generates $ X_k = D_k B_k M_k X + N_k $
        where: $M_k = I$ always, and $ B_k = I$ for now.
        $X_k$ has shape $(N_1, N_2)$, and $X$ has shape $(qN_1, qN_2)$.
    """

    lr_shape = get_lr_dims(image.shape, q, nb_lr_images)

    X = np.zeros(lr_shape)

    for k in range(nb_lr_images):
        M, B = 1, 1
        image_tmp = B * M * image
        noise = np.random.normal(0, noise_variance, X[k].shape)
        X[k] = image[::q, ::q] + noise
    return X


def get_hr_dims(shape, q):
    hr_dims = list(shape)
    del hr_dims[0]
    for i in range(2):
        hr_dims[i] *= q
    return tuple(hr_dims)

def data_fidelity_gradient(hr_image, lr_images, q):

    gradE = np.zeros_like(hr_image)

    
    for k in range(lr_images.shape[0]):
        # Upsampling here
        temp_hr = np.zeros_like(hr_image)
        for i in range(lr_images.shape[1]):
            for j in range(lr_images.shape[2]):
                temp_hr[q*i, q*j] = lr_images[k,i,j]
        gradE += hr_image - temp_hr
    gradE /= lr_images.shape[0]
    return gradE

def data_fidelity_gradient_2(hr_image, lr_images, q):

    gradE = np.zeros(hr_image.shape)

    #Downsampling HR image
    temp_lr = hr_image[::q, ::q]
    for k in range(lr_images.shape[0]):
        gradE += upsample(temp_lr - lr_images[k], q, hr_image.shape)
    #gradE /= lr_images.shape[0]
    return gradE
