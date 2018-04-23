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
