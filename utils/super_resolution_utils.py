import numpy as np

def get_lr_dims(shape, q, nb_lr_images):
    hr_shape = list(shape)
    lr_shape = hr_shape
    for i in range(2):
        lr_shape[i] = int(lr_shape[i]/q)
    lr_shape.insert(0, nb_lr_images)
    return tuple(lr_shape)

def createLRrSamples(image, q=2, nb_lr_images=8, noise_variance=10):
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

#pythran export upsample(float[][][][], int)
def upsample(lr_im, q):
    # Get new dimensions
    hr_dims = get_hr_dims(lr_im.shape, q)

    # Upsampling (D^T)
    temp_hr = np.zeros(hr_dims)
    #omp parallel for
    for i in range(lr_im.shape[0]):
        for j in range(lr_im.shape[1]):
            temp_hr[q*i, q*j] = lr_im[i,j]
    #temp_hr = 1*temp_hr # Unblurring (B^T)
    #temp_hr = 1*temp_hr # Unwarping (M^T)
    return temp_hr

#pythran export get_hr_dims((int, int, int), int)
def get_hr_dims(shape, q):
    hr_dims = list(shape)
    del hr_dims[0]
    for i in range(2):
        hr_dims[i] *= q
    return tuple(hr_dims)

#pythran export data_fidelity_gradient(float[][][], float[][][][], int)
def data_fidelity_gradient(hr_image, lr_images, q):

    gradE = np.zeros_like(hr_image)

    #omp parallel for reduction(:+gradE)
    for k in range(lr_images.shape[0]):
        # Upsampling here
        temp_hr = np.zeros_like(hr_image)
        for i in range(lr_images.shape[1]):
            for j in range(lr_images.shape[2]):
                temp_hr[q*i, q*j] = lr_images[k,i,j]
        gradE += hr_image - temp_hr
    gradE /= lr_images.shape[0]
    return gradE
